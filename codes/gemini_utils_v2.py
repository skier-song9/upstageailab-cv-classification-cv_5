import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
import timm
from torch.utils.data import Dataset
from types import SimpleNamespace
import yaml
import random
import numpy as np
import pandas as pd
from PIL import Image

def load_config(config_path='./config.yaml'):
    """.yaml 설정 파일 읽기

    :param str config_path: _description_, defaults to './config.yaml'
    :return _type_: _description_
    """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return SimpleNamespace(**cfg)

def set_seed(seed: int = 256):
    """랜덤성 제어 함수

    :param int seed: _description_, defaults to 256
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    """커스텀 데이터셋 클래스

    :param _type_ Dataset: _description_
    """
    def __init__(self, df:pd.DataFrame, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df.iloc[idx]
        # img = np.array(Image.open(os.path.join(self.path, name)))
        img = Image.open(os.path.join(self.path, name))
        if self.transform:
            # img = self.transform(image=img)['image']
            img = self.transform(image=np.array(img))['image']
        return img, target
    
### Getters
def get_activation(activation_option):
    ACTIVATIONS = {
        'None': None,
        'ReLU': nn.ReLU, # 음수는 0으로, 양수는 선형함수
        'LeakyReLU': nn.LeakyReLU, # 음수도 일부 통과 -> Dead ReLU 방지
        'ELU': nn.ELU, # 음수도 일부 통과 → 출력 평균이 0에 가깝도록 함으로써 학습 안정화 도움
        'SELU': nn.SELU, # ELU에 스케일링 계수를 곱해 신경망을 자기 정규화, 특정 조건 (예: fully connected, 특정 초기화, 특정 구조)에서만 자기 정규화 효과가 잘 발휘
        'GELU': nn.GELU, # 더 부드러운 비선형성, Transformer, BERT류
        'Tanh': nn.Tanh, # 완만한 sigmoid
        'PReLU': nn.PReLU, # 음수도 일부 통과 -> Dead ReLU 방지, 기울기를 학습함.
        'SiLU': nn.SiLU, # 더 부드러운 비선형성, EfficientNet, Swin Transformer 등
    }
    return ACTIVATIONS[activation_option]

class TimmWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        additional_options = {} # dropout 관련 옵션이 있는 모델의 경우
        for key in self.cfg.timm.keys():
            if key != "activation":
                additional_options[key] = self.cfg.timm[key]
        self.backbone = timm.create_model(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0, global_pool='avg',
            act_layer=get_activation(cfg.timm['activation']),
            **additional_options
        )
        self.dropout = nn.Dropout(p=cfg.custom_layer['drop'])
        self.activation = get_activation(cfg.custom_layer['activation'])()
        if cfg.custom_layer['head_type'] == "simple_dropout":
            self.classifier = nn.Sequential(
                self.dropout,
                nn.Linear(self.backbone.num_features, 17)
            )
        else :
            self.classifier = nn.Sequential(
                nn.Linear(self.backbone.num_features, 512),
                nn.BatchNorm1d(512),
                self.activation,
                self.dropout,
                nn.Linear(512, 17)
            )
        
        def weight_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if cfg.timm['activation'] in ['Tanh']:
                    # Xavier 초기화
                    init.xavier_uniform_(m.weight)
                else:
                    # He 초기화
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if cfg.fine_tuning == 'head':
            # backbone 파라미터를 freeze
            for param in self.backbone.parameters():
                param.requires_grad = False
            # classifier는 가중치 초기화
            self.classifier.apply(weight_init)
        elif cfg.fine_tuning == 'custom':
            # 직접 커스터마이징
            pass
        elif cfg.fine_tuning == 'scratch':
            self.apply(weight_init)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def get_timm_model(cfg):
    if hasattr(cfg, 'custom_layer') and cfg.custom_layer:
        return TimmWrapper(cfg).to(cfg.device)

    else: # timm 모델 구조 사용
        additional_options = {} # dropout 관련 옵션이 있는 모델의 경우
        for key in cfg.timm.keys():
            if key != "activation":
                additional_options[key] = cfg.timm[key]
        model = timm.create_model(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=17,
            act_layer=get_activation(cfg.timm['activation']),
            **additional_options
        )
        return model.to(cfg.device)

def get_criterion(cfg):
    CRITERIONS = {
        "CrossEntropyLoss" : nn.CrossEntropyLoss()
    }
    return CRITERIONS[cfg.criterion]

def get_optimizer(model, cfg):
    # SGD, RMSprop, Momentum, NAG, Adam, AdamW, NAdam, RAdam, Adafactor
    # optimizer_params = {k: v for k, v in vars(cfg.optimizer_params).items()}
    OPTIMIZERS = {
        'SGD': optim.SGD(model.parameters(), lr=cfg.lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, weight_decay=cfg.weight_decay),
        'Momentum': optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9),
        'NAG' : optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True),
        'Adam' : optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
        'AdamW': optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
        'NAdam': optim.NAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum_decay=4e-3),
        'RAdam': optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay),
        'Adafactor': optim.Adafactor(model.parameters(), lr=cfg.lr, beta2_decay=-0.8, d=1.0, weight_decay=cfg.weight_decay, maximize=False)
    }
    return OPTIMIZERS[cfg.optimizer_name]

def get_scheduler(optimizer, cfg, steps_per_epoch):
    # scheduler_params = {k: v for k, v in vars(cfg.scheduler_params).items()}
    # if cfg.scheduler_name == 'OneCycleLR':
    #     scheduler_params['steps_per_epoch'] = steps_per_epoch
    #     scheduler_params['epochs'] = cfg.epochs
    
    # StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
    SCHEDULERS = {
        'StepLR': lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1),
        'ExponentialLR': lr_scheduler.ExponentialLR(optimizer, gamma=0.1),
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0),
        'OneCycleLR': lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs),
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg.patience-5, min_lr=0),
    }
    return SCHEDULERS[cfg.scheduler_name]