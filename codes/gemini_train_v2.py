import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
import copy
import time
from types import SimpleNamespace
from sklearn.metrics import f1_score

import sys
sys.path.append(
	# 자신의 환경에 맞는 경로 설정
	"/data/ephemeral/home/upstageailab-cv-classification-cv_5/codes"
)

from gemini_augmentation_v2 import get_augmentation

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state_dict = None
        self.best_loss = None
        self.best_loss_epoch = 0
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss, epoch=0):
        if self.best_loss is None:
            #현재의 모델로 self.best_loss, self.best_model_state_dict 업데이트
            self.best_loss = val_loss
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
			#val_loss가 best_loss보다 좋을 때 > self.best_loss와 self.best_model 업데이트
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
			#val_loss가 더 안 좋을 때 > patience 증가하고 early stop 여부 확인
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights and self.best_model_state_dict is not None:
                    model.load_state_dict(self.best_model_state_dict)
                return True # ealy stopped
        return False # end with no early stop
    
    def restore_best(self, model):
        if self.best_loss is not None and self.best_model_state_dict is not None:
            print(f"Restore model_state_dict of which best_loss: {self.best_loss:.6f}")
            model.load_state_dict(self.best_model_state_dict)
            return True
        return False
	
class TrainModule():
	def __init__(self, model: torch.nn.Module, criterion, optimizer, scheduler, train_loader, valid_loader, cfg: SimpleNamespace, verbose:int =50, run=None):
		'''
		model, criterion, scheduler, train_loader, valid_loader 미리 정의해서 전달
		cfg : es_patience, epochs 등에 대한 hyperparameters를 namespace 객체로 입력
		'''
		required_attrs = ['scheduler_name','patience', 'epochs']
		for attr in required_attrs:
			assert hasattr(cfg, attr), f"AttributeError: There's no '{attr}' attribute in cfg."
		assert verbose > 0 and verbose < cfg.epochs, f"Logging frequency({verbose}) MUST BE smaller than EPOCHS({cfg.epochs}) and positive value."
		
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.cfg = cfg
		if getattr(cfg, "device", False):
			self.model.to(self.cfg.device)
		else:
			self.cfg.device = 'cpu'
		self.es = EarlyStopping(patience=self.cfg.patience)
		### list for plot
		self.train_losses_for_plot, self.val_losses_for_plot = [], []
		self.train_acc_for_plot, self.val_acc_for_plot = [], [] # classification
		self.train_f1_for_plot, self.val_f1_for_plot = [], [] # classification
		# logging frequency
		self.verbose = verbose
		# wandb run object
		self.run = run
		# Mixed Precision > 'cuda' device 에서만 가능하다.
		self.scaler = torch.amp.GradScaler(enabled=self.cfg.mixed_precision) # 기본적으로 FP16에 최적화되어 있습니다.
		self.epoch_counter = 0

	def training_step(self):
		# set train mode
		self.model.train()
		running_loss = 0.0
		correct = 0 # classification
		total = 0
		all_preds = []
		all_targets = []
		
		for train_x, train_y in self.train_loader: # batch training
			train_x, train_y = train_x.to(self.cfg.device), train_y.to(self.cfg.device)
			
			self.optimizer.zero_grad() # 이전 gradient 초기화

			# if self.cfg.mixed_precision: 
				# autocast 컨텍스트 매니저 사용 > # FP16을 사용해 메모리 사용량 감소
			with torch.amp.autocast(device_type='cuda', enabled=self.cfg.mixed_precision):
				outputs = self.model(train_x)
				loss = self.criterion(outputs, train_y)
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update() # 다음 반복을 위해 스케일 팩터를 업데이트

			# else:
			# 	outputs = self.model(train_x)
			# 	loss = self.criterion(outputs, train_y)
			# 	loss.backward() # backward pass
			# 	self.optimizer.step() # 가중치 업데이트

			if self.cfg.scheduler_name in ["OneCycleLR"]:
				self.scheduler.step()
			elif self.cfg.scheduler_name in ["CosineAnnealingWarmupRestarts"]:
				self.scheduler.step(self.epoch_counter)
			
			running_loss += loss.item() * train_y.size(0) # train_loss 
			_, predicted = torch.max(outputs, 1) # 가장 확률 높은 클래스 예측 # classification
			correct += (predicted == train_y).sum().item() # classification
			total += train_y.size(0) 

			all_preds.extend(predicted.cpu().numpy())
			all_targets.extend(train_y.cpu().numpy())

			# **********************************************
			# VRAM 부족 시: 각 배치 처리 후 GPU 캐시 비우기
			del train_x, train_y, outputs, loss # 사용된 변수 명시적 삭제
			torch.cuda.empty_cache()           # <-- 여기에 추가
			# **********************************************
			
		epoch_loss = running_loss / total # average loss of 1 epoch
		epoch_acc = 100 * correct / total # classification
		epoch_f1 = f1_score(all_targets, all_preds, average='macro') # classification
		return epoch_loss, epoch_acc, epoch_f1  # classification		
	
	def validation_step(self):
		if self.cfg.tta_dropout: 
			# inference 시에도 dropout을 유지하여 마치 앙상블하는 것 같은 효과를 준다.
			self.model.train()
		else:
			self.model.eval()  # 평가 모드
		self.model.eval()  # 평가 모드
		val_loss = 0
		correct = 0 # classification
		total = 0
		all_preds = []
		all_targets = []
		
		with torch.no_grad():  # gradient 계산 비활성화
			for val_x, val_y in self.valid_loader: # batch training
				val_x, val_y = val_x.to(self.cfg.device), val_y.to(self.cfg.device)
				
				# if self.cfg.mixed_precision: # FP16을 사용해 메모리 사용량 감소
				# autocast 컨텍스트 매니저 사용
				with torch.amp.autocast(device_type='cuda', enabled=self.cfg.mixed_precision):
					outputs = self.model(val_x)
					loss = self.criterion(outputs, val_y)
				# else:
				# 	outputs = self.model(val_x)
				# 	loss = self.criterion(outputs, val_y)
								
				val_loss += loss.item() * val_y.size(0)
				_, predicted = torch.max(outputs, 1) # classification
				correct += (predicted == val_y).sum().item() # classification
				total += val_y.size(0)

				all_preds.extend(predicted.cpu().numpy())
				all_targets.extend(val_y.cpu().numpy())

				# **********************************************
				# VRAM 부족 시: 각 배치 처리 후 GPU 캐시 비우기
				del val_x, val_y, outputs, loss # 사용된 변수 명시적 삭제
				torch.cuda.empty_cache()           # <-- 여기에 추가
				# **********************************************
		
		epoch_loss = val_loss / total # average loss of 1 epoch
		epoch_acc = 100 * correct / total # classification
		epoch_f1 = f1_score(all_targets, all_preds, average='macro') # classification
		return epoch_loss, epoch_acc, epoch_f1 # classification
	
	def update_transform(self, epoch):
		train_transforms, _, _, _ = get_augmentation(self.cfg, epoch)
		if self.cfg.online_augmentation:
			self.train_loader.dataset.transform = train_transforms[0]
		else:
			# ConcatDataset의 경우, 각 sub-dataset의 transform을 업데이트해야 합니다.
			for i, dataset in enumerate(self.train_loader.dataset.datasets):
				dataset.transform = train_transforms[i]

	def training_loop(self):
		# try:
		# reset loss list for plots
		self.train_losses_for_plot, self.val_losses_for_plot = [], []
		self.train_acc_for_plot, self.val_acc_for_plot = [], []
		self.train_f1_for_plot, self.val_f1_for_plot = [], []
		self.epoch_counter = 0
		epoch_timer = []
		done = False
		
		pbar = tqdm(total=self.cfg.epochs)
		while not done and self.epoch_counter<=self.cfg.epochs:
			self.update_transform(self.epoch_counter) # epoch에 따라 증강 기법을 바꾼다.
			st = time.time()
			self.epoch_counter += 1
			
			# train
			# train_loss = self.training_step() # regression
			train_loss, train_acc, train_f1 = self.training_step() # classification
			
			self.train_losses_for_plot.append(train_loss)
			self.train_acc_for_plot.append(train_acc) # classification
			self.train_f1_for_plot.append(train_f1) # classification

			# scheduler의 종류에 따라 val_loss를 전달하거나 그냥 step() 호출.
			if self.cfg.scheduler_name == "OneCycleLR":
				pass
			elif self.cfg.scheduler_name != "ReduceLROnPlateau":
				self.scheduler.step()

			if self.valid_loader is not None:
				# validation
				# val_loss = self.validation_step() # regression
				val_loss, val_acc, val_f1 = self.validation_step()  # classification
				self.val_losses_for_plot.append(val_loss)
				self.val_acc_for_plot.append(val_acc) # classification
				self.val_f1_for_plot.append(val_f1) # classification

				if self.cfg.scheduler_name == "ReduceLROnPlateau":
					self.scheduler.step(val_loss)

			epoch_timer.append(time.time() - st)
			pbar.update(1)
			
			if self.run is not None:
				# print('wandb logging...')
				epoch_log = {
					'train_loss': train_loss,
					'train_accuracy': train_acc,
					'train_f1': train_f1,
					'learning_rate': self.optimizer.param_groups[0]['lr'],
				}
				if self.valid_loader is not None:
					epoch_log['val_loss'] = val_loss
					epoch_log['val_accuracy'] = val_acc
					epoch_log['val_f1'] = val_f1

				# logging weights & gradients
				all_grads = []
				all_weights = []
				for param in self.model.parameters():
					if param.data is not None:
						all_weights.append(param.data.cpu().view(-1))
					if param.grad is not None:
						all_grads.append(param.grad.cpu().view(-1))
				if all_grads:
					epoch_log['weight/all'] = wandb.Histogram(torch.cat(all_weights))
				if all_weights:
					epoch_log['weights/all'] = wandb.Histogram(torch.cat(all_weights))
				
				self.run.log(epoch_log, step=self.epoch_counter) # wandb logging
			if self.epoch_counter == 1 or self.epoch_counter % self.verbose == 0:
				# self.verbose epoch마다 logging
				mean_time_spent = np.mean(epoch_timer)
				epoch_timer = [] # reset timer list
				# print(f"Epoch {self.epoch_counter}/{self.cfg.epochs} [Time: {mean_time_spent:.2f}s], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.8f}")
				if self.valid_loader is not None:
					print(f"Epoch {self.epoch_counter}/{self.cfg.epochs} [Time: {mean_time_spent:.2f}s], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.8f}\n Train ACC: {train_acc:.2f}%, Validation ACC: {val_acc:.2f}%\n Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}") # classification
				else:
					print(f"Epoch {self.epoch_counter}/{self.cfg.epochs} [Time: {mean_time_spent:.2f}s], Train Loss: {train_loss:.4f} | Train ACC: {train_acc:.2f}% | Train F1: {train_f1:.4f}") # classification
			if self.valid_loader is not None and self.es(self.model, val_loss, self.epoch_counter):
				# early stopped 된 경우 if 문 안으로 들어온다.
				done = True
		# except Exception as e:
		# 	print(e)
		# 	return False # training loop failed
		return True # training loop succeed
		
	def plot_loss(self, show:bool=False, savewandb:bool=True, savedir:str=None):
		"""loss, accuracy, f1-score에 대한 그래프 시각화 함수

		:param bool show: plt.show()를 실행할 건지, defaults to False
		:param bool savewandb: wandb logging에 plot을 시각화하여 저장할 건지, defaults to True
		:param str savedir: plot을 저장할 디렉토리를 설정, None이면 저장 안 함, defaults to None
		:return _type_: None
		"""
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(figsize=(6, 4))
		plt.plot(range(len(self.train_losses_for_plot)),self.train_losses_for_plot,color='blue',label='train_loss')
		plt.plot(range(len(self.val_losses_for_plot)),self.val_losses_for_plot,color='red',label='val_loss')
		plt.axhline(y=1e-3, color='red', linestyle='--', label='(Overfit)')
		plt.legend()
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Train/Validation Loss plot")
		if savedir is not None:
			if os.path.exists(savedir):
				os.makedirs(savedir, exist_ok=True)
			savepath = os.path.join(savedir, "loss_plot.png")
			plt.savefig(savepath)
			print(f"⚙️loss plot saved in {savepath}")
		if show:
			plt.show()
		if savewandb and self.run is not None:
			self.run.log({'loss_plot': wandb.Image(fig)}) # wandb
		plt.clf()
		
		# classification
		fig, ax = plt.subplots(figsize=(6, 4))
		plt.plot(range(len(self.train_acc_for_plot)),self.train_acc_for_plot,color='blue',label='train_acc')
		plt.plot(range(len(self.val_acc_for_plot)),self.val_acc_for_plot,color='red',label='val_acc')
		plt.axhline(y=99.0, color='red', linestyle='--', label='(99%)')
		plt.legend()
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy(%)")
		plt.title("Train/Validation Accuracy Plot")
		plt.grid()
		if savedir is not None:
			savepath = os.path.join(savedir, "accuracy_plot.png")
			plt.savefig(savepath)
			print(f"⚙️accuracy plot saved in {savepath}")
		if show:
			plt.show()
		if savewandb and self.run is not None:
			self.run.log({'accuracy_plot': wandb.Image(fig)}) # wandb
		plt.clf()

		# classification
		fig, ax = plt.subplots(figsize=(6, 4))
		plt.plot(range(len(self.train_f1_for_plot)),self.train_f1_for_plot,color='blue',label='train_f1')
		plt.plot(range(len(self.val_f1_for_plot)),self.val_f1_for_plot,color='red',label='val_f1')
		plt.axhline(y=0.99, color='red', linestyle='--', label='(0.99)')
		plt.legend()
		plt.xlabel("Epoch")
		plt.ylabel("F1-score")
		plt.title("Train/Validation F1-score Plot")
		plt.grid()
		if savedir is not None:
			savepath = os.path.join(savedir, "f1_plot.png")
			plt.savefig(savepath)
			print(f"⚙️f1 plot saved in {savepath}")
		if show:
			plt.show()
		if savewandb and self.run is not None:
			self.run.log({'f1_plot': wandb.Image(fig)}) # wandb
		plt.clf()
		return None
		
	def save_experiments(self, savepath=None):
		""""""
		save_dict = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'cfg': vars(self.cfg) # 나중에 로드해서 CFG = SimpleNamespace(**cfg)로 복원
		}
		if savepath is not None:
			dirpath = os.path.dirname(savepath)
			if os.path.exists(dirpath):
				os.makedirs(dirpath, exist_ok=True)
			torch.save(save_dict, f=savepath)
			return True
		return False
		