import sys
import os
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from types import SimpleNamespace
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import copy
import time
from PIL import Image
from zoneinfo import ZoneInfo
from datetime import datetime
import argparse
import traceback

#📢 project_root 설정 필수
project_root = '/data/ephemeral/home/upstageailab-cv-classification-cv_5'
sys.path.append(project_root)
from codes.gemini_utils_v3 import *
from codes.gemini_train_v3 import *
from codes.gemini_augmentation_v3 import *
from codes.gemini_evalute_v3 import *
from codes.gemini_main_v3 import (
    HARD_CLASSES, MODEL_B_CLASS_MAP, INV_MODEL_B_CLASS_MAP, create_model_a_reindex_maps,
    reindex_df_labels
)

def load_checkpoint_model(savepath=None):
    if os.path.exists(savepath):
        checkpoint = torch.load(savepath)
        cfg = SimpleNamespace(**checkpoint['cfg'])
        model = get_timm_model(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, cfg
    else:
        raise ValueError(f"{savepath} does not exists!")

MODEL_A_CLASS_MAP, INV_MODEL_A_CLASS_MAP = create_model_a_reindex_maps()

class TestTTAImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transforms_list):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transforms_list = transforms_list # TTA 변환 리스트 (ToTensorV2 포함)
    def __len__(self):
        return len(self.dataframe) * len(self.transforms_list) # 각 이미지당 TTA 수만큼 증가
    def __getitem__(self, idx):
        original_idx = idx // len(self.transforms_list)
        transform_idx = idx % len(self.transforms_list)

        img_id = self.dataframe.iloc[original_idx]['ID']
        img_path = os.path.join(self.img_dir, f"{img_id}")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV는 BGR, PIL은 RGB

        transform_func = self.transforms_list[transform_idx]
        augmented_image = transform_func(image=image)['image']
        return augmented_image, original_idx, transform_idx # 원본 이미지의 인덱스를 함께 반환하여 나중에 취합

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run deep learning training with specified configuration.")
        parser.add_argument(
            '--data',
            type=str,
            default='aug_data_500_new1', # 기본값 설정
            help='data directory'
        )
        parser.add_argument(
            '--sub',
            type=str,
            default='YYMMDDHHMM-ModelA-...ModelB-...', # 기본값 설정
            help='submission directory name'
        )
        parser.add_argument(
            '--tta',
            type=bool,
            default=True, # 기본값 설정
            help='submission directory name'
        )
        args = parser.parse_args()

        # model load
        ## 저장된 trainer를 로드하여 추론하는 경우 아래 함수로 로드
        
        submission_dir = os.path.join(project_root, args.data, "submissions", args.sub)
        model_a_path = os.path.join(submission_dir, "ModelA-" + args.sub + ".pth")
        model_b_path = os.path.join(submission_dir, "ModelB-" + args.sub + ".pth")
            
        model_a, cfg_a = load_checkpoint_model(
            savepath=model_a_path
        )
        model_b, cfg_b = load_checkpoint_model(
            savepath=model_a_path
        )

        # set TTA
        cfg_a.test_TTA = args.tta
        cfg_b.test_TTA = args.tta

        device = cfg_a.device

        ### Data Load
        df = pd.read_csv(os.path.join(cfg_a.data_dir, cfg_a.train_data))
        # --- 증강 설정 ---
        train_transforms_a, val_transform_a, val_tta_transform_a, test_tta_transform_a = get_augmentation(cfg_a, epoch=0)
        train_transforms_b, val_transform_b, val_tta_transform_b, test_tta_transform_b = get_augmentation(cfg_b, epoch=0)
        
        # 추론 파트
        print("===== Starting Hierarchical Inference =====")
        test_df = pd.read_csv(os.path.join(cfg_a.data_dir, "sample_submission.csv"))
        print("Step a) Predicting with Model A...")
        test_dataset_a = ImageDataset(test_df, os.path.join(cfg_a.data_dir, "test"), transform=val_transform_a)
        test_loader_a = DataLoader(test_dataset_a, batch_size=cfg_a.batch_size, shuffle=False, num_workers=8, pin_memory=True)    
        if cfg_a.test_TTA:
            print("Running TTA on test set...")
            tta_augs_a = [
                A.Compose([A.HorizontalFlip(p=1.0), test_tta_transform_a]), # 수평 반전
                A.Compose([A.VerticalFlip(p=1.0), test_tta_transform_a]),   # 수직 반전
                A.Compose([A.Transpose(p=1.0), test_tta_transform_a]),      # 대칭 (Transposition)
                A.Compose([A.Rotate(limit=(-10, 10), p=1.0), test_tta_transform_a]), # 미세한 회전
                test_tta_transform_a # 증강하지 않는 원본 이미지 변환 (마지막에 추가)
            ]
            test_tta_dataset_a = TestTTAImageDataset(
                dataframe=test_df,
                img_dir=os.path.join(cfg_a.data_dir,"test"),
                transforms_list=tta_augs_a
            )
            test_loader_a_tta = DataLoader(test_tta_dataset_a, batch_size=cfg_a.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            # preds_a = tta_predict(model_a, test_dataset_a, test_tta_transform_a, device, cfg_a, flag='test')
            if cfg_a.tta_dropout:
                model_a.train()
            else:
                model_a.eval()
            all_tta_outputs_a = torch.zeros(len(test_df), len(tta_augs_a), model_a.num_classes) # (원본 이미지 수, TTA 수, 클래스 수)
            with torch.no_grad():
                for images, original_indices, tta_indices in tqdm(test_loader_a_tta, desc="test TTA Prediction"):
                    images = images.to(device)
                    outputs = model_a(images)
                    probabilities = outputs.softmax(1).cpu()

                    # 배치 내의 각 결과에 대해 해당하는 원본 이미지 인덱스와 TTA 인덱스에 저장
                    for i in range(len(original_indices)):
                        orig_idx = original_indices[i].item()
                        tta_idx = tta_indices[i].item()
                        all_tta_outputs_a[orig_idx, tta_idx, :] = probabilities[i]
            # 각 원본 이미지별 TTA 결과 확률을 평균내고, 가장 높은 확률의 클래스를 선택
            avg_preds_a = torch.mean(all_tta_outputs_a, dim=1).numpy()
            preds_a = np.argmax(avg_preds_a, axis=1)
        else:
            print("Running inference on test set...")
            preds_a = predict(model_a, test_loader_a, device)

        pred_A_df = test_df.copy()
        pred_A_df['target'] = preds_a

        # b) Model A에서 Hard Class가 아닌 경우 결과 취합
        print("Step b) Collecting non-hard class predictions from Model A...")
        results_not_hard = pred_A_df[pred_A_df['target'] != 0].copy()
        results_not_hard['target'] = results_not_hard['target'].apply(lambda x: INV_MODEL_A_CLASS_MAP.get(x))

        # c) Model A에서 Hard Class로 예측된 이미지를 Model B로 예측
        print("Step c) Predicting hard classes with Model B...")
        ids_for_model_b = pred_A_df[pred_A_df['target'] == 0]
        pred_B_df = pd.DataFrame()
        if not ids_for_model_b.empty:
            test_dataset_b = ImageDataset(ids_for_model_b, os.path.join(cfg_b.data_dir, "test"), transform=val_transform_b)
            test_loader_b = DataLoader(test_dataset_b, batch_size=cfg_b.batch_size, shuffle=False, num_workers=8, pin_memory=True)    
            if cfg_b.test_TTA:
                print("Running TTA on test set...")
                tta_augs_b = [
                    A.Compose([A.HorizontalFlip(p=1.0), test_tta_transform_b]), # 수평 반전
                    A.Compose([A.VerticalFlip(p=1.0), test_tta_transform_b]),   # 수직 반전
                    A.Compose([A.Transpose(p=1.0), test_tta_transform_b]),      # 대칭 (Transposition)
                    A.Compose([A.Rotate(limit=(-10, 10), p=1.0), test_tta_transform_b]), # 미세한 회전
                    test_tta_transform_b # 증강하지 않는 원본 이미지 변환 (마지막에 추가)
                ]
                test_tta_dataset_b = TestTTAImageDataset(
                    dataframe=ids_for_model_b,
                    img_dir=os.path.join(cfg_b.data_dir,"test"),
                    transforms_list=tta_augs_b
                )
                test_loader_b_tta = DataLoader(test_tta_dataset_b, batch_size=cfg_b.batch_size, shuffle=False, num_workers=8, pin_memory=True)
                # preds_a = tta_predict(model_a, test_dataset_a, test_tta_transform_a, device, cfg_a, flag='test')
                if cfg_b.tta_dropout:
                    model_b.train()
                else:
                    model_b.eval()
                all_tta_outputs_b = torch.zeros(len(ids_for_model_b), len(tta_augs_b), model_b.num_classes) # (원본 이미지 수, TTA 수, 클래스 수)
                with torch.no_grad():
                    for images, original_indices, tta_indices in tqdm(test_loader_b_tta, desc="test TTA Prediction"):
                        images = images.to(device)
                        outputs = model_b(images)
                        probabilities = outputs.softmax(1).cpu()

                        # 배치 내의 각 결과에 대해 해당하는 원본 이미지 인덱스와 TTA 인덱스에 저장
                        for i in range(len(original_indices)):
                            orig_idx = original_indices[i].item()
                            tta_idx = tta_indices[i].item()
                            all_tta_outputs_b[orig_idx, tta_idx, :] = probabilities[i]
                # 각 원본 이미지별 TTA 결과 확률을 평균내고, 가장 높은 확률의 클래스를 선택
                avg_preds_b = torch.mean(all_tta_outputs_b, dim=1).numpy()
                preds_b = np.argmax(avg_preds_b, axis=1)
            else:
                print("Running inference on test set...")
                preds_b = predict(model_b, test_loader_b, device)
            pred_B_df = ids_for_model_b.copy()
            pred_B_df['target'] = preds_b
        else:
            print("No images were classified as the hard class by Model A.")

        # d) Model B 결과 Re-Indexing
        print("Step d) Re-indexing Model B predictions...")
        if not pred_B_df.empty:
            pred_B_df['target'] = pred_B_df['target'].apply(lambda x: INV_MODEL_B_CLASS_MAP.get(x))

        # e) 최종 결과 취합
        print("Step e) Combining results and creating submission file...")
        final_submission_df = pd.read_csv(os.path.join(cfg_a.data_dir, "sample_submission.csv"))
        final_submission_df = final_submission_df.set_index('ID')
        if not results_not_hard.empty:
            final_submission_df.loc[results_not_hard['ID'], 'target'] = results_not_hard.set_index('ID')['target']
        if not pred_B_df.empty:
            final_submission_df.loc[pred_B_df['ID'], 'target'] = pred_B_df.set_index('ID')['target']
            
        final_submission_df.reset_index(inplace=True)

        # --- 제출 ---
        submission_path = os.path.join(submission_dir, f"{args.sub}.csv")
        final_submission_df.to_csv(submission_path, index=False)
        print(f"📢 Submission file saved to {submission_path}")

        ### prediction class별 개수
        try:
            class_counts = final_submission_df['target'].value_counts().sort_index()
            class_counts = class_counts.reset_index(drop=False)
            meta = pd.read_csv(os.path.join(cfg_a.data_dir, "meta.csv"))
            meta_dict = zip(meta['target'], meta['class_name'])
            meta_dict = dict(meta_dict)
            targets_class = list(map(lambda x: meta_dict[x], class_counts['target']))
            class_counts['meta'] = targets_class
            class_counts.to_csv(os.path.join(submission_dir, "submission_class_distribution.csv"), index=False)
        except Exception as e:
            print(e)

    finally:
        pass


