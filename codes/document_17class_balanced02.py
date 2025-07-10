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
            # 의료문서 특화 개인정보 마스킹 시뮬레이션
            A.OneOf([
                # 수평 마스킹 (환자 정보 마스킹)
                A.CoarseDropout(
                    max_holes=3, max_height=15, max_width=200,
                    min_holes=1, min_height=10, min_width=100,
                    fill_value=0, p=1.0
                ),
                # 의료기관명 마스킹
                A.CoarseDropout(
                    max_holes=2, max_height=25, max_width=150,
                    min_holes=1, min_height=15, min_width=80,
                    fill_value=0, p=1.0
                ),
                # 부분 마스킹 (진료내용 등)
                A.CoarseDropout(
                    max_holes=5, max_height=20, max_width=120,
                    min_holes=2, min_height=10, min_width=60,
                    fill_value=0, p=1.0
                ),
                A.NoOp(p=1.0),
            ], p=0.6),
            
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
    
    def get_rotation_robust_augmentation(self):
        """회전/뒤집힘 강건성 증강 (Test dataset 대응)"""
        return A.Compose([
            # 다양한 회전 시뮬레이션 (Test 환경)
            A.OneOf([
                A.Rotate(limit=(-180, 180), border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
                A.RandomRotate90(p=1.0),
                A.Transpose(p=1.0),
                A.VerticalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
                A.NoOp(p=1.0),
            ], p=0.4),
            
            # 노이즈 추가 (Test dataset 특성)
            A.OneOf([
                A.GaussNoise(var_limit=(0.001, 0.05), mean=0, p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.3),
            
            # 제목 부분 손실 시뮬레이션 (짤림 현상 대응)
            A.OneOf([
                A.CoarseDropout(max_holes=2, max_height=30, max_width=100, 
                               min_holes=1, min_height=10, min_width=50, 
                               fill_value=255, p=1.0),
                A.NoOp(p=1.0),
            ], p=0.2),
        ])

    def get_train_transform(self):
        """훈련용 변환 (제목 영역 보존 + Test 환경 대응)"""
        return A.Compose([
            # 1단계: 크기 조정 (제목 영역 위치 확정)
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
            
            # 2단계: 다양한 증강 전략 적용
            A.OneOf([
                # 의료문서 특화 증강 (제목 영역 최소 변형)
                self.get_medical_document_augmentation(),
                # gemini 스타일 증강 (일반적인 변형)
                self.get_gemini_style_augmentation(),
                # 회전 강건성 증강 (Test 환경 시뮬레이션)
                self.get_rotation_robust_augmentation(),
                A.NoOp(),
            ], p=0.85),  # 85% 확률로 증강 적용
            
            # 3단계: 정규화
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    
    def get_val_transform(self):
        """검증용 변환"""
        return A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

class OutpatientVsDischargeDetector(nn.Module):
    """통원 vs 퇴원 구분 특화 모듈"""
    
    def __init__(self, feature_dim):
        super().__init__()
        
        # 1. 미세 문자 패턴 감지기 (통원/퇴원 구분)
        self.character_pattern_detector = nn.Sequential(
            # 고해상도 패턴 감지
            nn.Conv2d(feature_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 수직 스트로크 감지 (通 vs 退의 차이)
            nn.Conv2d(512, 256, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 수평 스트로크 감지
            nn.Conv2d(256, 128, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 세밀한 패턴 추출
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 2. 점(·) 패턴 감지기 (입·퇴원 vs 통원확인서 구분)
        self.dot_pattern_detector = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 1),
            nn.ReLU(),
            # 매우 작은 커널로 점 패턴 감지
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # 3. 문서 레이아웃 복잡도 분석기
        self.layout_complexity_analyzer = nn.Sequential(
            # 테이블 라인 감지
            nn.Conv2d(feature_dim, 256, 7, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 수평 라인 강화 감지
            nn.Conv2d(256, 128, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 수직 라인 강화 감지  
            nn.Conv2d(128, 64, (7, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 복잡도 측정
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 4. Standard-10-view 제목 영역 집중 모듈 (회전 대응)
        self.standard_10_view_attention = nn.ModuleDict({
            # 원본 방향 (0도)
            'view_0': self._create_title_detector(feature_dim, 'horizontal'),
            # 90도 회전 (세로 방향)
            'view_90': self._create_title_detector(feature_dim, 'vertical'),
            # 180도 회전 (뒤집힌 가로)
            'view_180': self._create_title_detector(feature_dim, 'horizontal_flipped'),
            # 270도 회전 (뒤집힌 세로)
            'view_270': self._create_title_detector(feature_dim, 'vertical_flipped'),
            # 대각선 방향들
            'view_45': self._create_title_detector(feature_dim, 'diagonal_1'),
            'view_135': self._create_title_detector(feature_dim, 'diagonal_2'),
            'view_225': self._create_title_detector(feature_dim, 'diagonal_3'),
            'view_315': self._create_title_detector(feature_dim, 'diagonal_4'),
            # 약간의 기울어짐
            'view_tilt_1': self._create_title_detector(feature_dim, 'slight_tilt_1'),
            'view_tilt_2': self._create_title_detector(feature_dim, 'slight_tilt_2'),
        })
        
        # 10-view 통합 어텐션 생성기
        self.view_aggregator = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 방향별 가중치 학습
        self.orientation_weights = nn.Parameter(torch.ones(10) / 10)
    
    def _create_title_detector(self, feature_dim, orientation_type):
        """방향별 제목 감지기 생성"""
        if orientation_type in ['horizontal', 'horizontal_flipped']:
            # 가로 방향 제목 감지 (일반적인 문서)
            return nn.Sequential(
                nn.Conv2d(feature_dim, 256, (3, 7), padding=(1, 3)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, (1, 5), padding=(0, 2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        elif orientation_type in ['vertical', 'vertical_flipped']:
            # 세로 방향 제목 감지 (90도 회전)
            return nn.Sequential(
                nn.Conv2d(feature_dim, 256, (7, 3), padding=(3, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, (5, 1), padding=(2, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        elif orientation_type.startswith('diagonal'):
            # 대각선 방향 제목 감지
            return nn.Sequential(
                nn.Conv2d(feature_dim, 256, 5, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        else:  # slight_tilt 등
            # 약간 기울어진 제목 감지
            return nn.Sequential(
                nn.Conv2d(feature_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. 문자 패턴 분석
        char_patterns = self.character_pattern_detector(x)
        
        # 2. 점 패턴 분석 (상위 20% 영역에서만)
        title_region = x[:, :, :h//5, :]
        dot_patterns = self.dot_pattern_detector(title_region)
        # 전체 이미지 크기로 패딩
        dot_patterns = F.pad(dot_patterns, (0, 0, 0, h - h//5), value=0)
        
        # 3. 레이아웃 복잡도 분석
        complexity_score = self.layout_complexity_analyzer(x)
        complexity_map = complexity_score.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        # 4. Standard-10-view 제목 영역 어텐션 (회전 강건성)
        view_attentions = []
        
        # 각 방향별로 제목 영역 감지
        for i, (view_name, detector) in enumerate(self.standard_10_view_attention.items()):
            # 상위 영역에서 제목 감지 (방향에 따라 다른 영역 설정)
            if 'vertical' in view_name:
                # 세로 방향: 좌측 20% 영역
                region = x[:, :, :, :w//5]
                view_attention = detector(region)
                # 전체 크기로 패딩
                view_attention = F.pad(view_attention, (0, w - w//5, 0, 0), value=0)
            elif 'diagonal' in view_name:
                # 대각선 방향: 코너 영역들
                if view_name == 'view_45':  # 좌상단
                    region = x[:, :, :h//3, :w//3]
                    view_attention = detector(region)
                    view_attention = F.pad(view_attention, (0, w - w//3, 0, h - h//3), value=0)
                elif view_name == 'view_135':  # 우상단
                    region = x[:, :, :h//3, -w//3:]
                    view_attention = detector(region)
                    view_attention = F.pad(view_attention, (w - w//3, 0, 0, h - h//3), value=0)
                elif view_name == 'view_225':  # 좌하단
                    region = x[:, :, -h//3:, :w//3]
                    view_attention = detector(region)
                    view_attention = F.pad(view_attention, (0, w - w//3, h - h//3, 0), value=0)
                else:  # view_315: 우하단
                    region = x[:, :, -h//3:, -w//3:]
                    view_attention = detector(region)
                    view_attention = F.pad(view_attention, (w - w//3, 0, h - h//3, 0), value=0)
            else:
                # 가로 방향 및 기타: 상위 15% 영역
                region = x[:, :, :h//7, :]  # 더 정확한 제목 영역
                view_attention = detector(region)
                # 전체 크기로 패딩
                view_attention = F.pad(view_attention, (0, 0, 0, h - h//7), value=0)
            
            view_attentions.append(view_attention)
        
        # 10개 뷰를 하나로 통합
        stacked_views = torch.cat(view_attentions, dim=1)  # [B, 10, H, W]
        
        # 방향별 가중치 적용
        weighted_views = stacked_views * self.orientation_weights.view(1, 10, 1, 1)
        
        # 통합 어텐션 생성
        title_attention = self.view_aggregator(weighted_views)
        
        # 5. 종합 특성 생성
        # 통원 패턴 강화 (char_patterns가 높으면 통원 가능성)
        outpatient_enhancement = char_patterns * 2.0
        # 점 패턴 강화 (dot_patterns가 높으면 입퇴원 가능성)
        discharge_enhancement = dot_patterns * 3.0
        # 복잡도 기반 강화 (복잡하면 진료확인서 가능성)
        complexity_enhancement = complexity_map * 1.5
        # 제목 영역 강화
        title_enhancement = title_attention * 2.5
        
        # 최종 특성 맵 생성
        enhanced_features = x * (1.0 + 
                                outpatient_enhancement + 
                                discharge_enhancement + 
                                complexity_enhancement + 
                                title_enhancement)
        
        return enhanced_features, {
            'char_patterns': char_patterns.mean().item(),
            'dot_patterns': dot_patterns.mean().item(), 
            'complexity': complexity_score.mean().item(),
            'title_attention': title_attention.mean().item()
        }

class Class3vs7SpecializedClassifier(nn.Module):
    """클래스 3 vs 7 특화 분류기"""
    
    def __init__(self, feature_dim, num_classes=17):
        super().__init__()
        
        # 주 분류기 (17클래스)
        self.main_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim//2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim//2, num_classes)
        )
        
        # 클래스 3 vs 7 이진 분류기
        self.binary_3vs7_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim//2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim//2, feature_dim//4),
            nn.ReLU(),
            nn.Linear(feature_dim//4, 2),  # 클래스 3 vs 7
            nn.Softmax(dim=1)
        )
        
        # 통원/퇴원 구분 점수
        self.outpatient_vs_discharge_scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim//4),
            nn.ReLU(),
            nn.Linear(feature_dim//4, feature_dim//8),
            nn.ReLU(),
            nn.Linear(feature_dim//8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        main_logits = self.main_classifier(x)
        binary_3vs7_probs = self.binary_3vs7_classifier(x)
        outpatient_score = self.outpatient_vs_discharge_scorer(x)
        
        return main_logits, binary_3vs7_probs, outpatient_score

class GeminiStyleModel(nn.Module):
    """통원/퇴원 구분 특화 문서 분류 모델"""
    
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k_384', num_classes=17):
        super().__init__()
        
        # 백본 모델
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # 통원/퇴원 구분 모듈
        self.outpatient_detector = OutpatientVsDischargeDetector(self.backbone.num_features)
        
        # 글로벌 풀링
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 특화 분류기
        self.classifier = Class3vs7SpecializedClassifier(self.backbone.num_features, num_classes)
        
        # 후처리 규칙
        self.post_processing_enabled = True
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # 백본 특성 추출
        features = self.backbone(x)
        
        # 통원/퇴원 패턴 분석 및 특성 강화
        enhanced_features, pattern_info = self.outpatient_detector(features)
        
        # 글로벌 풀링
        pooled_features = self.global_pool(enhanced_features)
        pooled_features = pooled_features.flatten(1)
        
        # 분류
        main_logits, binary_3vs7_probs, outpatient_score = self.classifier(pooled_features)
        
        # 후처리 적용 (추론 시)
        if not self.training and self.post_processing_enabled:
            main_logits = self.apply_post_processing(main_logits, binary_3vs7_probs, 
                                                   outpatient_score, pattern_info)
        
        return main_logits, binary_3vs7_probs, outpatient_score, pattern_info
    
    def apply_post_processing(self, main_logits, binary_3vs7_probs, outpatient_score, pattern_info):
        """클래스 3 vs 7 혼동 해결 후처리"""
        enhanced_logits = main_logits.clone()
        
        # 클래스 3과 7의 확률
        class_3_prob = torch.softmax(main_logits, dim=1)[:, 3]  # 입퇴원확인서
        class_7_prob = torch.softmax(main_logits, dim=1)[:, 7]  # 진료확인서
        
        # 3 vs 7 혼동 상황 감지
        confusion_mask = (class_3_prob > 0.2) & (class_7_prob > 0.2)
        
        if confusion_mask.any():
            # 점 패턴이 강하면 클래스 3 (입·퇴원확인서) 강화
            dot_pattern_strong = pattern_info['dot_patterns'] > 0.3
            if dot_pattern_strong:
                enhanced_logits[confusion_mask, 3] += 1.0
                enhanced_logits[confusion_mask, 7] -= 0.5
            
            # 통원 패턴이 강하고 복잡도가 높으면 클래스 7 (진료확인서) 강화
            outpatient_pattern_strong = pattern_info['char_patterns'] > 0.4
            high_complexity = pattern_info['complexity'] > 0.6
            if outpatient_pattern_strong and high_complexity:
                enhanced_logits[confusion_mask, 7] += 1.2
                enhanced_logits[confusion_mask, 3] -= 0.8
            
            # 이진 분류기 결과 반영
            binary_class7_prob = binary_3vs7_probs[:, 1]  # 클래스 7 확률
            high_binary_7_prob = binary_class7_prob > 0.7
            enhanced_logits[confusion_mask & high_binary_7_prob, 7] += 0.8
            enhanced_logits[confusion_mask & high_binary_7_prob, 3] -= 0.6
        
        return enhanced_logits

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
        
        # 균형잡힌 클래스 가중치 (의료문서 편향 제거)
        self.criterion = nn.CrossEntropyLoss()  # 기본 손실함수
        
        # Mixup/Cutmix 설정
        self.use_mixup = getattr(cfg, 'use_mixup', True)
        self.use_cutmix = getattr(cfg, 'use_cutmix', True)
        self.mixup_alpha = getattr(cfg, 'mixup_alpha', 0.2)
        self.cutmix_alpha = getattr(cfg, 'cutmix_alpha', 1.0)
        self.mixup_prob = getattr(cfg, 'mixup_prob', 0.5)  # Mixup과 Cutmix 적용 확률
        
        # gemini 스타일 최적화기
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # gemini 스타일 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.lr * 0.01
        )
        
        # 조기 종료
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_model_state = None
        
        # Mixed Precision (gemini 스타일)
        self.scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision else None
    
    def train_epoch(self):
        """한 에포크 훈련 (Mixup/Cutmix 적용)"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixup/Cutmix 적용 여부 결정
            use_aug = False
            labels_a, labels_b, lam = None, None, None  # 변수 초기화
            
            if self.use_mixup or self.use_cutmix:
                if np.random.random() < self.mixup_prob:
                    use_aug = True
                    if self.use_mixup and self.use_cutmix:
                        # 50% 확률로 Mixup 또는 Cutmix 선택
                        if np.random.random() < 0.5:
                            images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                            aug_type = 'mixup'
                        else:
                            images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                            aug_type = 'cutmix'
                    elif self.use_mixup:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                        aug_type = 'mixup'
                    else:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                        aug_type = 'cutmix'
            
            # Mixed Precision (gemini 스타일)
            if self.scaler:
                with torch.cuda.amp.autocast():
                    model_outputs = self.model(images)
                    main_logits = model_outputs[0]  # 주 분류기 출력
                    binary_3vs7_probs = model_outputs[1]  # 이진 분류기 출력
                    outpatient_score = model_outputs[2]  # 통원/퇴원 점수
                    
                    # 주 손실 계산
                    if use_aug:
                        main_loss = mixup_criterion(self.criterion, main_logits, labels_a, labels_b, lam)
                    else:
                        main_loss = self.criterion(main_logits, labels)
                    
                    # 보조 손실 (클래스 3 vs 7 특화)
                    binary_labels = torch.zeros_like(labels)
                    binary_labels[labels == 7] = 1  # 클래스 7을 1로
                    binary_labels[labels == 3] = 0  # 클래스 3을 0으로
                    
                    # 클래스 3, 7에 해당하는 샘플만 이진 분류 손실 계산
                    class_3_7_mask = (labels == 3) | (labels == 7)
                    if class_3_7_mask.sum() > 0:
                        if use_aug:
                            # Mixup 적용 시 양쪽 레이블 모두 고려
                            binary_loss_a = F.cross_entropy(binary_3vs7_probs[class_3_7_mask], binary_labels[class_3_7_mask])
                            binary_labels_b = torch.zeros_like(labels_b)
                            binary_labels_b[labels_b == 7] = 1
                            binary_labels_b[labels_b == 3] = 0
                            class_3_7_mask_b = (labels_b == 3) | (labels_b == 7)
                            if class_3_7_mask_b.sum() > 0:
                                binary_loss_b = F.cross_entropy(binary_3vs7_probs[class_3_7_mask_b], binary_labels_b[class_3_7_mask_b])
                                binary_loss = lam * binary_loss_a + (1 - lam) * binary_loss_b
                            else:
                                binary_loss = binary_loss_a
                        else:
                            binary_loss = F.cross_entropy(binary_3vs7_probs[class_3_7_mask], binary_labels[class_3_7_mask])
                    else:
                        binary_loss = torch.tensor(0.0, device=self.device)
                    
                    # 총 손실
                    loss = main_loss + 0.3 * binary_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                model_outputs = self.model(images)
                main_logits = model_outputs[0]
                
                if use_aug:
                    loss = mixup_criterion(self.criterion, main_logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(main_logits, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 예측 레이블 계산 (주 분류기 출력 사용)
            _, predicted = torch.max(main_logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            if use_aug:
                # Mixup/Cutmix 적용시 더 높은 가중치를 가진 레이블 사용
                actual_labels = labels_a if lam > 0.5 else labels_b
                all_labels.extend(actual_labels.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
            # VRAM 최적화 (gemini 스타일)
            del images, main_logits, loss
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
        """검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        model_outputs = self.model(images)
                        main_logits = model_outputs[0]
                        loss = self.criterion(main_logits, labels)
                else:
                    model_outputs = self.model(images)
                    main_logits = model_outputs[0]
                    loss = self.criterion(main_logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(main_logits.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, f1, accuracy, all_preds, all_labels
    
    def train(self):
        """전체 훈련 과정"""
        print(f"🚀 균형잡힌 훈련 시작 - {self.cfg.epochs}에포크")
        
        for epoch in range(self.cfg.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.cfg.epochs} ---")
            
            # 훈련
            train_loss, train_f1 = self.train_epoch()
            
            # 검증
            val_loss, val_f1, val_acc, val_preds, val_labels = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 성능 모델 저장
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"🎉 새로운 최고 성능! F1: {val_f1:.4f}")
                
                # 클래스별 성능 분석
                if epoch % 10 == 0:
                    self._analyze_class_performance(val_labels, val_preds)
            else:
                self.patience_counter += 1
            
            # 조기 종료
            if self.patience_counter >= self.cfg.patience:
                print(f"조기 종료: {self.cfg.patience} 에포크 동안 성능 향상 없음")
                break
        
        # 최고 성능 모델로 복원
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"최고 성능 모델로 복원 (F1: {self.best_val_f1:.4f})")
        
        return self.best_val_f1
    
    def _analyze_class_performance(self, true_labels, pred_labels):
        """클래스별 성능 분석"""
        per_class_f1 = f1_score(true_labels, pred_labels, average=None)
        
        # 클래스 3 vs 7 혼동 분석
        print("\n🏥 클래스 3 vs 7 혼동 분석:")
        class_3_indices = [i for i, label in enumerate(true_labels) if label == 3]
        class_7_indices = [i for i, label in enumerate(true_labels) if label == 7]
        
        if class_3_indices:
            class_3_preds = [pred_labels[i] for i in class_3_indices]
            class_3_to_7_errors = sum(1 for pred in class_3_preds if pred == 7)
            class_3_accuracy = sum(1 for pred in class_3_preds if pred == 3) / len(class_3_preds)
            print(f"  클래스 3 (입퇴원확인서): 정확도 {class_3_accuracy:.3f}")
            print(f"  클래스 3→7 오분류: {class_3_to_7_errors}/{len(class_3_indices)} ({class_3_to_7_errors/len(class_3_indices)*100:.1f}%)")
        
        if class_7_indices:
            class_7_preds = [pred_labels[i] for i in class_7_indices]
            class_7_to_3_errors = sum(1 for pred in class_7_preds if pred == 3)
            class_7_accuracy = sum(1 for pred in class_7_preds if pred == 7) / len(class_7_preds)
            print(f"  클래스 7 (진료확인서): 정확도 {class_7_accuracy:.3f}")
            print(f"  클래스 7→3 오분류: {class_7_to_3_errors}/{len(class_7_indices)} ({class_7_to_3_errors/len(class_7_indices)*100:.1f}%)")
        
        # 비문서 클래스 성능 확인
        non_document_classes = {
            0: "account_number (손글씨)",
            2: "car_dashboard", 
            5: "driver_lisence",
            8: "national_id_card",
            9: "passport",
            15: "vehicle_registration_certificate",
            16: "vehicle_registration_plate"
        }
        
        print("\n🚗 비문서 클래스 성능:")
        for cls_id, cls_name in non_document_classes.items():
            if cls_id < len(per_class_f1):
                print(f"  클래스 {cls_id:2d} ({cls_name}): F1={per_class_f1[cls_id]:.3f}")

def apply_class3_vs_7_postprocessing(predictions, confidence_scores=None):
    """클래스 3 vs 7 혼동 해결 후처리
    
    분석 결과 기반 규칙:
    - "통원" 키워드가 포함된 문서는 클래스 7 (진료확인서)
    - "입·퇴원" 키워드가 포함된 문서는 클래스 3 (입퇴원확인서)  
    - 혼동 시 진료확인서가 더 일반적이므로 클래스 7로 보정
    """
    enhanced_predictions = predictions.copy()
    
    # 후처리 적용 통계
    class_3_to_7_changes = 0
    
    # 혼동이 예상되는 경우의 후처리 규칙
    for i, pred in enumerate(predictions):
        if confidence_scores is not None:
            conf = confidence_scores[i]
            
            # 낮은 신뢰도의 클래스 3 예측을 클래스 7로 변경
            # 이유: "통원확인서", "통원치료서" 등이 클래스 3으로 잘못 분류되는 경우가 많음
            if pred == 3 and conf < 0.6:  # 신뢰도 임계값 낮춤
                enhanced_predictions[i] = 7
                class_3_to_7_changes += 1
    
    if class_3_to_7_changes > 0:
        print(f"🔄 후처리 적용: 클래스 3→7 변경 {class_3_to_7_changes}개")
    
    return enhanced_predictions

def predict_test(model, test_loader, device):
    """테스트 예측 (후처리 포함)"""
    model.eval()
    
    # BatchNorm을 평가 모드로 강제 설정 (배치 크기 1 대응)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = True
    
    predictions = []
    image_ids = []
    confidence_scores = []
    
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            model_outputs = model(images)
            main_logits = model_outputs[0]  # 주 분류기 출력만 사용
            
            # 소프트맥스로 신뢰도 계산
            probs = torch.softmax(main_logits, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            predictions.extend(preds.cpu().numpy())
            confidence_scores.extend(max_probs.cpu().numpy())
            image_ids.extend(ids)
    
    # 클래스 3 vs 7 후처리 적용
    enhanced_predictions = apply_class3_vs_7_postprocessing(predictions, confidence_scores)
    
    return enhanced_predictions, image_ids

def main():
    parser = argparse.ArgumentParser(description="17클래스 문서 분류 - 균형잡힌 버전 (Mixup/Cutmix 적용)")
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
    data_dir = "/data/ephemeral/home/upstageailab-cv-classification-cv_5/aug_data_500_new1"
    
    print(f"🚀 통원/퇴원 구분 특화 분류 시작 (OCR 없는 고급 패턴 인식)")
    print(f"🖥️  Device: {cfg.device}")
    print(f"🔧 Model: {cfg.model_name}")
    print(f"📐 Image Size: {cfg.image_size}")
    print(f"💾 Data Dir: {data_dir}")
    print(f"🎯 Mixup: {cfg.use_mixup} (alpha={cfg.mixup_alpha})")
    print(f"🎯 Cutmix: {cfg.use_cutmix} (alpha={cfg.cutmix_alpha})")
    print(f"🎯 Aug Prob: {cfg.mixup_prob}")
    print(f"🔍 통원(通院) vs 퇴원(退院) 문자 패턴 감지 활성화")
    print(f"⚫ 점(·) 패턴 감지 시스템 활성화 (입·퇴원확인서)")
    print(f"📊 문서 레이아웃 복잡도 분석 활성화")
    print(f"📝 제목 영역 집중 어텐션 활성화")
    print(f"🔄 회전/뒤집힘/노이즈 강건성 증강 적용")
    print(f"🎯 클래스 3 vs 7 특화 이진 분류기 포함")
    
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
    print(f"📊 테스트 데이터: {len(test_df)}개 (aug_data_1000_new1 데이터셋)")
    
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
    
    # 데이터로더 (gemini 스타일) - drop_last=True로 배치 크기 1 방지
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True  # 마지막 배치가 1개 샘플일 때 제거
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        drop_last=True  # 검증에서도 마지막 배치 제거
    )
    
    # gemini 스타일 모델
    model = GeminiStyleModel(cfg.model_name, num_classes=17)
    model.to(cfg.device)
    
    print(f"📊 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    trainer = BalancedDocumentTrainer(model, train_loader, val_loader, cfg)
    trainer.train()
    
    # 테스트 예측
    print(f"\n🔮 테스트 예측")
    test_dataset = BalancedDocumentDataset(test_df, test_dir, val_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        drop_last=False  # 테스트에서는 모든 샘플 예측 필요
    )
    
    predictions, image_ids = predict_test(model, test_loader, cfg.device)
    
    # 결과 저장
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    submissions_dir = os.path.join(data_dir, "submissions")
    output_path = os.path.join(submissions_dir, f"balanced_output_10view_characterspecific_{timestamp}.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result_df = pd.DataFrame({
        'ID': image_ids,
        'target': predictions
    })
    
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✅ 균형잡힌 예측 결과 저장: {output_path}")
    print(f"📊 예측 분포:")
    print(result_df['target'].value_counts().sort_index())
    
    # 비문서 클래스 예측 분석
    non_document_classes = [0, 2, 5, 8, 9, 15, 16]
    non_document_predictions = result_df[result_df['target'].isin(non_document_classes)]
    print(f"\n🚗 비문서 클래스 예측 수: {len(non_document_predictions)}")
    
    print(f"\n🎉 균형잡힌 문서 분류 완료!")

if __name__ == "__main__":
    main()