"""
17클래스 문서 분류 - 균형잡힌 버전
gemini_main_v2_1.py 분석 결과를 반영한 개선된 코드

주요 개선사항:
1. gemini 스타일 데이터 증강 (morphological, affine 중심)
2. 단순화된 모델 구조 (TIMM 래퍼 기반)
3. 균형잡힌 클래스 처리 (의료문서 편향 제거)
4. 범용 정규화 (0.5, 0.5, 0.5)
5. 비문서 클래스 특화 처리
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
import random
import math
from torch.optim.lr_scheduler import _LRScheduler
warnings.filterwarnings('ignore')

class CosineAnnealingWarmupRestarts(_LRScheduler):
    # ref : https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1: # 초기 상태
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # 웜업 단계: 학습률이 min_lr에서 max_lr로 선형적으로 증가합니다.
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            # 코사인 어닐링 단계: 학습률이 max_lr에서 min_lr로 코사인 함수 형태로 감소합니다.
            return [base_lr + (self.max_lr - base_lr) * \
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / \
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2 
                    for base_lr in self.base_lrs]
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Mixup과 Cutmix 구현
def mixup_data(x, y, alpha=0.2):
    """Mixup 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 손실 함수"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """Cutmix 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # 이미지 크기
    _, _, H, W = x.shape
    
    # 마스크 영역 계산
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # 중심점 랜덤 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 경계 계산
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 실제 lambda 값 조정
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # Cutmix 적용
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# gemini 스타일 Morphological 연산
class Morphological(A.ImageOnlyTransform):
    def __init__(self, scale=(1, 3), operation="dilation", always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale = scale
        self.operation = operation

    def apply(self, img, **params):
        k = np.random.randint(self.scale[0], self.scale[1] + 1)
        kernel = np.ones((k, k), np.uint8)
        
        if self.operation == "dilation":
            return cv2.dilate(img, kernel, iterations=1)
        elif self.operation == "erosion":
            return cv2.erode(img, kernel, iterations=1)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")

class BalancedDocumentTransforms:
    """gemini 스타일 균형잡힌 데이터 증강"""
    
    def __init__(self, image_size=384):
        self.image_size = image_size
    
    def get_gemini_style_augmentation(self):
        """gemini 코드 스타일 증강 (EDA 기반)"""
        return A.Compose([
            # 색상 조정 (gemini 스타일)
            A.ColorJitter(brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=0.8),
            
            # 기하학적 변형 (gemini 핵심)
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 30),
                shear=(-5, 5),
                cval=255,  # 흰색 패딩
                p=0.9
            ),
            
            # 반전 변환 (gemini 스타일)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            
            # 텍스트 특화 처리 (비문서 클래스를 위한 핵심)
            A.OneOf([
                Morphological(scale=(1, 3), operation="dilation", p=1.0),
                Morphological(scale=(1, 3), operation="erosion", p=1.0),
                A.NoOp(p=1.0),
            ], p=0.4),
            
            # 최소한의 블러 (gemini 스타일)
            A.OneOf([
                A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                A.Blur(blur_limit=(3, 9), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.3),  # 확률 대폭 감소
            
            # 약간의 노이즈 (gemini 스타일)
            A.GaussNoise(var_limit=(0.0025, 0.1), p=0.3),  # 강도 감소
        ])
    
    def get_medical_document_augmentation(self):
        """의료문서 특화 증강 (진료확인서 vs 입퇴원확인서 오분류 방지)"""
        return A.Compose([
            # 의료문서 특화 개인정보 마스킹 시뮬레이션 (영향력 감소)
            A.OneOf([
                A.CoarseDropout(
                    max_holes=2, max_height=10, max_width=150,
                    min_holes=1, min_height=8, min_width=80,
                    fill_value=0, p=1.0
                ),
                A.CoarseDropout(
                    max_holes=1, max_height=20, max_width=120,
                    min_holes=1, min_height=12, min_width=70,
                    fill_value=0, p=1.0
                ),
                A.CoarseDropout(
                    max_holes=4, max_height=15, max_width=100,
                    min_holes=2, min_height=8, min_width=50,
                    fill_value=0, p=1.0
                ),
                A.NoOp(p=1.0),
            ], p=0.4), # 확률 감소
            
            # 의료문서 특화 스캔 품질 저하 시뮬레이션
            A.OneOf([
                A.CLAHE(clip_limit=(2.0, 4.0), p=1.0),  # 대비 강화
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=1.0
                ),
                A.NoOp(p=1.0),
            ], p=0.4),
            
            # 제목 영역 보존을 위한 최소 변형 (상위 25% 영역 보호)
            A.OneOf([
                A.Affine(
                    translate_percent={'x': (-0.01, 0.01), 'y': (0.0, 0.01)},  # 세로 이동 최소화 (제목 보호)
                    rotate=(-0.5, 0.5),  # 회전 각도 최소화
                    shear=(-0.5, 0.5),   # 기울기 최소화
                    cval=255,
                    p=1.0
                ),
                A.NoOp(p=1.0),
            ], p=0.5),  # 확률도 낮춤
            
            # 텍스트 패턴 강화
            A.OneOf([
                Morphological(scale=(1, 2), operation="dilation", p=1.0),
                Morphological(scale=(1, 2), operation="erosion", p=1.0),
                A.NoOp(p=1.0),
            ], p=0.3),
        ])
    
    def get_train_transform(self):
        """훈련용 변환 (제목 영역 보존 우선 + gemini 스타일)"""
        return A.Compose([
            # 1단계: 크기 조정 (제목 영역 위치 확정)
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1.0),
            
            # 2단계: 제목 영역 보존 우선 증강 (상위 25% 보호)
            A.OneOf([
                # 의료문서 특화 증강 (제목 영역 최소 변형)
                self.get_medical_document_augmentation(),
                # gemini 스타일 증강 (회전/변형 포함)
                self.get_gemini_style_augmentation(),
                A.NoOp(),
            ], p=0.85),  # 85% 확률로 증강 적용
            
            # 3단계: 정규화
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    
    def get_val_transform(self):
        """검증용 변환"""
        return A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1.0),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

class Standard10ViewAttention(nn.Module):
    """Standard 10-view Attention for Robust Feature Extraction"""
    def __init__(self, feature_dim, crop_scale=0.875):
        super().__init__()
        self.crop_scale = crop_scale
        self.attention_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        crop_h = int(height * self.crop_scale)
        crop_w = int(width * self.crop_scale)

        # Define 5 crop regions (center, four corners)
        crops = [
            x[:, :, (height - crop_h) // 2:(height + crop_h) // 2, (width - crop_w) // 2:(width + crop_w) // 2], # center
            x[:, :, 0:crop_h, 0:crop_w], # top-left
            x[:, :, 0:crop_h, width - crop_w:width], # top-right
            x[:, :, height - crop_h:height, 0:crop_w], # bottom-left
            x[:, :, height - crop_h:height, width - crop_w:width] # bottom-right
        ]

        # Add horizontally flipped versions to create 10 views
        flipped_crops = [torch.flip(c, [3]) for c in crops]
        all_views = crops + flipped_crops

        context_vectors = []
        for view in all_views:
            # Flatten spatial dimensions
            features = view.reshape(batch_size, channels, -1) 
            
            # Calculate attention scores
            attention_scores = self.attention_fc(features.transpose(1, 2)).squeeze(-1)
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # Apply attention
            context_vector = torch.bmm(attention_probs.unsqueeze(1), features.transpose(1, 2)).squeeze(1)
            context_vectors.append(context_vector)

        # Average the context vectors from all views
        final_context_vector = torch.stack(context_vectors).mean(dim=0)
        
        return final_context_vector

class GeminiStyleModel(nn.Module):
    """강화된 제목 인식 의료문서 모델"""
    
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k_384', num_classes=17):
        super().__init__()
        
        # gemini 스타일 백본 (글로벌 풀링 비활성화)
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # 10-view 어텐션 모듈
        self.attention = Standard10ViewAttention(self.backbone.num_features)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """gemini 스타일 가중치 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        attended_features = self.attention(features)
        logits = self.classifier(attended_features)
        return logits

class BalancedDocumentDataset(Dataset):
    """균형잡힌 문서 데이터셋"""
    
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        # 클래스명 매핑
        self.class_names = {
            0: "account_number", 1: "application_for_payment_of_pregnancy_medical_expenses", 
            2: "car_dashboard", 3: "confirmation_of_admission_and_discharge", 4: "diagnosis",
            5: "driver_lisence", 6: "medical_bill_receipts", 7: "medical_outpatient_certificate",
            8: "national_id_card", 9: "passport", 10: "payment_confirmation",
            11: "pharmaceutical_receipt", 12: "prescription", 13: "resume",
            14: "statement_of_opinion", 15: "vehicle_registration_certificate", 16: "vehicle_registration_plate"
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # gemini 스타일 단일 폴더 구조에 맞춰 수정
        image_path = os.path.join(self.image_dir, row['ID'])
        
        # 이미지 로드 (gemini 스타일)
        try:
            # PIL로 로드 후 numpy 배열로 변환 (gemini 스타일)
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'L':  # Grayscale
                img = img.convert('RGB')
            elif img.mode == 'P':  # Palette mode
                img = img.convert('RGB')
            
            img = np.array(img)  # PIL RGB는 이미 RGB 순서
        except Exception as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            img = np.ones((384, 384, 3), dtype=np.uint8) * 255
        
        # 변환 적용
        if self.transform:
            try:
                img = self.transform(image=img)['image']
            except Exception as e:
                print(f"변환 오류: {e}")
                img = torch.ones(3, 384, 384)
        
        if self.is_test:
            return img, row['ID']
        else:
            return img, row['target']

class BalancedDocumentTrainer:
    """균형잡힌 문서 분류 훈련 클래스"""
    
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.device
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.use_mixup = getattr(cfg, 'use_mixup', True)
        self.use_cutmix = getattr(cfg, 'use_cutmix', True)
        self.mixup_alpha = getattr(cfg, 'mixup_alpha', 0.2)
        self.cutmix_alpha = getattr(cfg, 'cutmix_alpha', 1.0)
        self.mixup_prob = getattr(cfg, 'mixup_prob', 0.5)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 스케줄러는 train() 메소드에서 훈련 스텝 수에 맞춰 동적으로 생성
        self.scheduler = None
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_model_state = None
        
        self.scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision else None
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            use_aug = False
            if self.use_mixup or self.use_cutmix:
                if np.random.random() < self.mixup_prob:
                    use_aug = True
                    if self.use_mixup and self.use_cutmix:
                        if np.random.random() < 0.5:
                            images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                        else:
                            images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                    elif self.use_mixup:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                    else:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if use_aug:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_aug:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # 스케줄러 업데이트 (매 배치마다)
            self.scheduler.step()

            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            if use_aug:
                actual_labels = labels_a if lam > 0.5 else labels_b
                all_labels.extend(actual_labels.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})
            
            del images, outputs, loss
            if use_aug:
                del labels_a, labels_b
            else:
                del labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, f1, accuracy, all_preds, all_labels
    
    def train(self):
        # 스케줄러 생성 (훈련 스텝 수에 맞춰)
        total_steps = len(self.train_loader) * self.cfg.epochs
        warmup_steps = 1 #int(total_steps * 0.1)
        
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.lr,
            min_lr=self.cfg.lr * 0.01,
            warmup_steps=warmup_steps,
            gamma=0.9
        )

        print(f"🚀 균형잡힌 훈련 시작 - {self.cfg.epochs}에포크")
        print(f"Scheduler: CosineAnnealingWarmupRestarts (Total steps: {total_steps}, Warmup steps: {warmup_steps})")

        for epoch in range(self.cfg.epochs):
            print(f"--- Epoch {epoch+1}/{self.cfg.epochs} ---")
            
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_f1, val_acc, val_preds, val_labels = self.validate()
            
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"🎉 새로운 최고 성능! F1: {val_f1:.4f}")
                
                if epoch % 10 == 0:
                    self._analyze_class_performance(val_labels, val_preds)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.cfg.patience:
                print(f"조기 종료: {self.cfg.patience} 에포크 동안 성능 향상 없음")
                break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"최고 성능 모델로 복원 (F1: {self.best_val_f1:.4f})")
        
        return self.best_val_f1
    
    def _analyze_class_performance(self, true_labels, pred_labels):
        """클래스별 성능 분석"""
        per_class_f1 = f1_score(true_labels, pred_labels, average=None)
        
        non_document_classes = {
            0: "account_number (손글씨)", 2: "car_dashboard", 5: "driver_lisence",
            8: "national_id_card", 9: "passport", 15: "vehicle_registration_certificate",
            16: "vehicle_registration_plate"
        }
        
        print("🚗 비문서 클래스 성능:")
        for cls_id, cls_name in non_document_classes.items():
            if cls_id < len(per_class_f1):
                print(f"  클래스 {cls_id:2d} ({cls_name}): F1={per_class_f1[cls_id]:.3f}")

def predict_test(model, test_loader, device):
    """테스트 예측"""
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)
    
    return predictions, image_ids

def main():
    parser = argparse.ArgumentParser(description="17클래스 문서 분류 - 균형잡힌 버전 (Mixup/Cutmix 적용)")
    parser.add_argument('--data', type=str, default='aug_data_500_new1')
    parser.add_argument('--model', type=str, default='convnext_base.fb_in22k_ft_in1k_384')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use Mixup augmentation')
    parser.add_argument('--use_cutmix', action='store_true', default=True, help='Use Cutmix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Cutmix alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='Probability of applying Mixup/Cutmix')
    args = parser.parse_args()
    
    # 설정
    cfg = SimpleNamespace(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-5,
        image_size=args.image_size,
        patience=args.patience,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        mixed_precision=True,
        random_seed=256,
        # Mixup/Cutmix 설정
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob
    )
    
    # 데이터 디렉토리 설정 (gemini_main_v2_1.py와 동일한 데이터셋 사용)
    data_dir = f"/data/ephemeral/home/upstageailab-cv-classification-cv_5/{args.data}"
    
    print(f"🚀 의료문서 특화 분류 시작 (gemini 스타일 + Mixup/Cutmix)")
    print(f"🖥️  Device: {cfg.device}")
    print(f"🔧 Model: {cfg.model_name}")
    print(f"📐 Image Size: {cfg.image_size}")
    print(f"💾 Data Dir: {data_dir}")
    print(f"🎯 Mixup: {cfg.use_mixup} (alpha={cfg.mixup_alpha})")
    print(f"🎯 Cutmix: {cfg.use_cutmix} (alpha={cfg.cutmix_alpha})")
    print(f"🎯 Aug Prob: {cfg.mixup_prob}")
    print(f"🏥 진료/입퇴원확인서 오분류 방지 시스템 활성화")
    
    # 시드 설정 (gemini 스타일)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    
    # 데이터 로드
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    print(f"📊 훈련 데이터: {len(train_df)}개")
    print(f"📊 테스트 데이터: {len(test_df)}개 (aug_data_500_new1 데이터셋)")
    
    # 데이터 분할
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=cfg.random_seed,
        stratify=train_df['target']
    )
    
    # gemini 스타일 변환
    transform_manager = BalancedDocumentTransforms(cfg.image_size)
    train_transform = transform_manager.get_train_transform()
    val_transform = transform_manager.get_val_transform()
    
    # 데이터셋
    train_dataset = BalancedDocumentDataset(train_data, train_dir, train_transform)
    val_dataset = BalancedDocumentDataset(val_data, train_dir, val_transform)
    
    # 데이터로더 (gemini 스타일)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # gemini 스타일 모델
    model = GeminiStyleModel(cfg.model_name, num_classes=17)
    model.to(cfg.device)
    
    print(f"📊 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    trainer = BalancedDocumentTrainer(model, train_loader, val_loader, cfg)
    trainer.train()
    
    # 테스트 예측
    print(f"🔮 테스트 예측")
    test_dataset = BalancedDocumentDataset(test_df, test_dir, val_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    predictions, image_ids = predict_test(model, test_loader, cfg.device)
    
    # 결과 저장
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    submissions_dir = os.path.join(data_dir, "submissions", "improved_header")
    output_path = os.path.join(submissions_dir, f"balanced_medical_output_{timestamp}.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_df = pd.DataFrame({
        'ID': image_ids,
        'target': predictions
    })
    
    result_df.to_csv(output_path, index=False)
    
    print(f"✅ 균형잡힌 예측 결과 저장: {output_path}")
    print(f"📊 예측 분포:")
    print(result_df['target'].value_counts().sort_index())
    
    # 비문서 클래스 예측 분석
    non_document_classes = [0, 2, 5, 8, 9, 15, 16]
    non_document_predictions = result_df[result_df['target'].isin(non_document_classes)]
    print(f"🚗 비문서 클래스 예측 수: {len(non_document_predictions)}")
    
    # 비문서 클래스 예측 분석
    hard_document_classes = [3,4,7,14]
    hard_document_predictions = result_df[result_df['target'].isin(hard_document_classes)]
    print(f"🚗 고난도 클래스 예측 수: {len(hard_document_predictions)}")

    # print(f"🎉 균형잡힌 문서 분류 완료!")

if __name__ == "__main__":
    """
    권장 설정 (30 에포크):
    cd /data/ephemeral/home/upstageailab-cv-classification-cv_5/codes
    python improved_balance_classifier.py \
        --data aug_data_1000_new1 \
        --model convnextv2_base.fcmae_ft_in22k_in1k_384 \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-4 \
        --image_size 384 \
        --patience 4 \
        --use_mixup \
        --use_cutmix \
        --mixup_alpha 0.2 \
        --cutmix_alpha 1.0 \
        --mixup_prob 0.7
    빠른 테스트 (15 에포크):
    python document_17class_balanced.py \
        --model convnext_base.fb_in22k_ft_in1k_384 \
        --epochs 15 \
        --batch_size 16 \
        --patience 8
    CUDA 메모리 부족시:
    python document_17class_balanced.py \
        --model convnext_base.fb_in22k_ft_in1k_384 \
        --epochs 15 \
        --batch_size 8 \
        --image_size 320
    """
    main()