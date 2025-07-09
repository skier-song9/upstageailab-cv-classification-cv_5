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

# 💡 HIERARCHICAL CLASSIFICATION CONSTANTS
HARD_CLASSES = {3, 4, 7, 14}
MODEL_B_CLASS_MAP = {val: i for i, val in enumerate(sorted(list(HARD_CLASSES)))}
INV_MODEL_B_CLASS_MAP = {v: k for k, v in MODEL_B_CLASS_MAP.items()}

def create_model_a_reindex_maps(total_classes=17):
    """Model A를 위한 re-indexing 맵을 생성합니다."""
    model_a_map = {c: 0 for c in HARD_CLASSES}
    current_new_idx = 1
    for i in range(total_classes):
        if i not in HARD_CLASSES:
            model_a_map[i] = current_new_idx
            current_new_idx += 1
    inv_model_a_map = {v: k for k, v in model_a_map.items()}
    return model_a_map, inv_model_a_map

MODEL_A_CLASS_MAP, INV_MODEL_A_CLASS_MAP = create_model_a_reindex_maps()

def reindex_df_labels(df, mapping):
    """주어진 매핑에 따라 데이터프레임의 'target' 레이블을 재정의합니다."""
    df_reindexed = df.copy()
    df_reindexed['original_target'] = df_reindexed['target']
    df_reindexed['target'] = df_reindexed['target'].apply(lambda x: mapping.get(x))
    return df_reindexed

def run_training_cycle(train_df, val_df, cfg, run, train_transforms, val_transform):
    augmented_ids, val_augmented_ids = [], []
    # 클래스 불균형 해소를 위한 이미지 offline 증강
    if hasattr(cfg, 'class_imbalance') and cfg_a.class_imbalance:
        new_augmented_ids, augmented_labels = augment_class_imbalance(cfg, train_df)
        augmented_ids.extend(new_augmented_ids)
        imb_aug_df = pd.DataFrame({"ID": new_augmented_ids, "target": augmented_labels})
        train_df = pd.concat([train_df, imb_aug_df], ignore_index=True).reset_index(drop=True)
    try:
        # validation 데이터를 offline으로 eda 증강을 적용
        if val_df is not None and cfg_a.val_TTA:
            new_val_augmented_ids, augmented_labels = augment_validation(cfg, val_df)
            val_augmented_ids.extend(new_val_augmented_ids)
            val_aug_df = pd.DataFrame({"ID": new_val_augmented_ids, "target": augmented_labels})
            val_df = pd.concat([val_df, val_aug_df], ignore_index=True).reset_index(drop=True)

        # Sampler 설정
        sampler = None
        shuffle = True
        if cfg_a.weighted_random_sampler:
            targets = train_df['target'].values
            class_counts = np.bincount(targets)
            class_weights = 1. / class_counts
            weights = class_weights[targets]
            g = get_generator(cfg)
            sampler = WeightedRandomSampler(weights, len(weights), generator=g)
            shuffle = False

        # Dataset 생성
        if cfg_a.online_augmentation:
            train_dataset = ImageDataset(train_df, os.path.join(cfg_a.data_dir, "train"), transform=train_transforms[0])
        else:
            datasets = [ImageDataset(train_df, os.path.join(cfg_a.data_dir, "train"), transform=t) for t in train_transforms]
            train_dataset = ConcatDataset(datasets)
        
        val_loader = None
        if val_df is not None:
            val_dataset = ImageDataset(val_df, os.path.join(cfg_a.data_dir, "train"), transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=cfg_a.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if cfg_a.online_aug['mixup']:
            train_collate = lambda batch: mixup_collate_fn(batch, num_classes=17, alpha=0.4)
        elif cfg_a.online_aug['cutmix']:
            train_collate = lambda batch: cutmix_collate_fn(batch, num_classes=17, alpha=0.4)
        else:
            train_collate = None
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=cfg_a.batch_size, sampler=sampler, shuffle=shuffle, num_workers=8, pin_memory=True, collate_fn=train_collate)

        trainer = TrainModule(
            train_df,
            train_loader=train_loader,
            valid_loader=val_loader,
            cfg=cfg,
            verbose=1,
            run=run
        )

        # 학습
        train_result = trainer.training_loop()
        trainer.es.restore_best(trainer.model)
        if not train_result:
            raise ValueError("Failed to train model...")
    except Exception as e:
        if augmented_ids:
            delete_offline_augmented_images_multithreaded(cfg, augmented_ids, num_threads=12)
            augmented_ids = []
        if val_augmented_ids:
            delete_offline_augmented_images_multithreaded(cfg, val_augmented_ids, num_threads=12)
            val_augmented_ids = []
        traceback.print_exc()
    return trainer, augmented_ids, val_augmented_ids, val_df, val_loader

def train_hierarchical_model(model_type, cfg, df, run, train_transforms, val_transform, val_tta_transform):
    """
    계층적 분류의 각 모델(A 또는 B)을 학습시킵니다.
    Cross-validation과 일반 학습을 모두 지원합니다.
    """
    print(f"===== Starting Training for Model {model_type} =====")
    # 1. 데이터 준비 (필터링 및 리인덱싱)
    if model_type == 'A':
        reindex_map = MODEL_A_CLASS_MAP
        num_classes = len(set(reindex_map.values()))
        train_df_filtered = reindex_df_labels(df, reindex_map)
    elif model_type == 'B':
        reindex_map = MODEL_B_CLASS_MAP
        num_classes = len(reindex_map)
        train_df_filtered = df[df['target'].isin(HARD_CLASSES)].reset_index(drop=True)
        train_df_filtered = reindex_df_labels(train_df_filtered, reindex_map)
    else:
        raise ValueError("model_type must be 'A' or 'B'")
    cfg_a.num_classes = num_classes

    train_df_filtered = train_df_filtered[['ID','target']]

    # 2. 학습 실행 (CV 또는 단일 분할)
    if cfg_a.n_folds >= 3:
        # Cross-validation
        folds_es, folds_val_f1 = [], []
        skf = StratifiedKFold(n_splits=cfg_a.n_folds, shuffle=True, random_state=cfg_a.random_seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df_filtered, train_df_filtered['target'])):
            trainer = None
            fold_aug_ids, fold_val_aug_ids = [], []
            train_losses_for_plot, val_losses_for_plot = [], []
            train_acc_for_plot, val_acc_for_plot = [], []
            train_f1_for_plot, val_f1_for_plot = [], []
            try:
                print(f"--- Model {model_type}, FOLD {fold+1} ---")
                train_fold_df, val_fold_df = train_df_filtered.iloc[train_idx], train_df_filtered.iloc[val_idx]
                
                trainer, fold_aug_ids, fold_val_aug_ids, val_aug_df, val_loader = run_training_cycle(
                    train_fold_df, val_fold_df, cfg, run=None, train_transforms=train_transforms, val_transform=val_transform
                )
                train_losses_for_plot.append(trainer.train_losses_for_plot)
                train_acc_for_plot.append(trainer.train_acc_for_plot)
                train_f1_for_plot.append(trainer.train_f1_for_plot)
                val_losses_for_plot.append(trainer.val_losses_for_plot)
                val_acc_for_plot.append(trainer.val_acc_for_plot)
                val_f1_for_plot.append(trainer.val_f1_for_plot)
                folds_es.append(trainer.es.best_loss_epoch)
                
                # (CV에서는 평가 메트릭만 간단히 로깅)
                _, val_f1 = do_validation(
                    val_aug_df, trainer.model, val_loader, None, cfg, run=None, show=False,
                    savepath=os.path.join(cfg_a.submission_dir, f"val_confusion_matrix{'_TTA' if cfg_a.val_TTA else ''}_Model-{model_type}_Fold{fold}.png"),
                    meta_dict=INV_MODEL_A_CLASS_MAP if model_type=='A' else INV_MODEL_B_CLASS_MAP
                )
                folds_val_f1.append(val_f1)
                del trainer
            finally:
                if fold_aug_ids:
                    delete_offline_augmented_images_multithreaded(cfg, fold_aug_ids, num_threads=12)
                    fold_aug_ids = []
                if fold_val_aug_ids:
                    delete_offline_augmented_images_multithreaded(cfg, fold_val_aug_ids, num_threads=12)
                    fold_val_aug_ids = []

        plot_cross_validation(train_losses_for_plot, val_losses_for_plot, "Loss", cfg, show=False, model_type=model_type)
        plot_cross_validation(train_acc_for_plot, val_acc_for_plot, "Accuracy", cfg, show=False, model_type=model_type)
        plot_cross_validation(train_f1_for_plot, val_f1_for_plot, "F1-score", cfg, show=False, model_type=model_type)
        best_epoch = int(np.mean(folds_es))
        cfg_a.epochs = best_epoch
        print(f"📢 Model {model_type} CV Done. Avg F1: {np.mean(folds_val_f1):.5f}, Best Epoch for final training: {best_epoch}")

        # CV 후 전체 데이터로 최종 모델 학습
        final_aug_ids, final_val_aug_ids = []
        try:
            final_trainer, final_val_aug_ids, _, _ = run_training_cycle(
                train_df_filtered, None, cfg, run, train_transforms, val_transform
            )
            final_trainer.save_experiments(savepath=os.path.join(cfg.submission_dir, f'Model{model_type}-{next_run_name}.pth'))
        finally:
            if final_aug_ids:
                delete_offline_augmented_images_multithreaded(cfg, final_aug_ids, num_threads=12)
                final_aug_ids = []
            if val_augmented_ids:
                delete_offline_augmented_images_multithreaded(cfg, final_val_aug_ids, num_threads=12)
                val_augmented_ids = []
        
        return final_trainer

    else:
        # No Cross-validation
        try:
            print("- Train/Validation 분할")
            train_split_df, val_split_df = train_test_split(
                train_df_filtered, test_size=cfg_a.val_split_ratio, random_state=cfg_a.random_seed, stratify=train_df_filtered['target']
            )
            print(f"    - train_split_df: {train_split_df.shape}, val_split_df: {val_split_df.shape}")
            
            print("- Run Training Cycle")
            trainer, aug_ids, val_aug_ids, val_aug_df, val_loader = run_training_cycle(
                train_split_df, val_split_df, cfg, run, train_transforms, val_transform
            )
            ### Save Model
            trainer.save_experiments(savepath=os.path.join(cfg.submission_dir, f'Model{model_type}-{next_run_name}.pth'))
            ## 학습 결과 시각화 저장.
            trainer.plot_loss(
                show=False,
                savewandb=cfg.wandb['log'],
                savedir=cfg.submission_dir,
                model_type=model_type
            )
            ### Evaluate
            val_preds, val_f1 = do_validation(
                df=val_aug_df, 
                model=trainer.model, 
                # data=val_dataset_raw if cfg.val_TTA else val_loader, # online validation TTA
                data = val_loader, # offline validation TTA
                transform_func=val_tta_transform, 
                cfg=cfg, 
                run=run, 
                show=False, 
                savepath=os.path.join(cfg.submission_dir, f"Model{model_type}_val_confusion_matrix{'_TTA' if cfg.val_TTA else ''}.png"),
                meta_dict=INV_MODEL_A_CLASS_MAP if model_type=='A' else INV_MODEL_B_CLASS_MAP
            )
            print("📢 Validation F1-score:",val_f1)
            # Save incorrect validation results
            try:
                save_validation_images(val_split_df, val_preds, cfg, images_per_row=5, show=False, model_type=model_type)
            except:
                print("⚠️Saving incorrect validation results Failed...")
        finally:
            if aug_ids:
                delete_offline_augmented_images_multithreaded(cfg, aug_ids, num_threads=12)
                aug_ids = []
            if val_aug_ids:
                delete_offline_augmented_images_multithreaded(cfg, val_aug_ids, num_threads=12)
                val_aug_ids = []
        return trainer


if __name__ == "__main__":
    try:
        # python 파일 실행할 때 config.yaml 파일 이름을 입력받아서 설정 파일을 지정한다.
        parser = argparse.ArgumentParser(description="Run deep learning training with specified configuration.")
        parser.add_argument(
            '--config',
            type=str,
            default='config.yaml', # 기본값 설정
            help='Name of the configuration YAML file (e.g., config.yaml, experiment_A.yaml)'
        )
        parser.add_argument(
            '--config2', # 2-stage 모델을 위한 config
            type=str,
            default='config2.yaml', # 기본값 설정
            help='Name of the configuration YAML file (e.g., config.yaml, experiment_A.yaml)'
        )

        args = parser.parse_args()

        # Yaml 파일 읽기
        cfg_a = load_config(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
        )
        cfg_b = load_config(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config2)
        )
        # 랜덤성 제어
        set_seed(cfg_a.random_seed)

        # device 설정
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        print("⚙️ Device :",device)
        cfg_a.device = device
        cfg_b.device = device
        CURRENT_TIME = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%y%m%d%H%M")
        print(f"⌚ 실험 시간: {CURRENT_TIME}")

        # W&B 설정
        # 증강 기법 문자열 생성 로직 개선
        aug_str_parts = ""
        if cfg_a.online_augmentation:
            aug_str_parts += "on"
            if cfg_a.dynamic_augmentation and cfg_a.dynamic_augmentation['enabled']:
                aug_str_parts += "daug"                
            else:
                # 일반 augmentation 기법들만 나열
                aug_str_parts += "aug"
        else:
            aug_str_parts += "offaug"

        next_run_name = (
            f"{CURRENT_TIME}-ModelA-"
            f"{cfg_a.model_name}-"
            f"opt_{cfg_a.optimizer_name}-"
            f"sch_{cfg_a.scheduler_name}-"
            f"img{cfg_a.image_size}-"
            f"es{cfg_a.patience}-"
            f"{aug_str_parts}-"  # 개선된 증강 문자열
            f"cv{cfg_a.n_folds}-"
            f"clsaug_{1 if cfg_a.class_imbalance else 0}-"
            f"vTTA_{1 if cfg_a.val_TTA else 0}-"
            f"tTTA_{1 if cfg_a.test_TTA else 0}-"
            f"MP_{1 if cfg_a.mixed_precision else 0}"
        )
        next_run_name2 = (
            f"{CURRENT_TIME}-ModelB-"
            f"{cfg_b.model_name}-"
            f"opt_{cfg_b.optimizer_name}-"
            f"sch_{cfg_b.scheduler_name}-"
            f"img{cfg_b.image_size}-"
            f"es{cfg_b.patience}-"
            f"{'mixup' if cfg_b.online_aug['mixup'] else 'cutmix'}-"  
            f"cv{cfg_b.n_folds}-"
            f"clsaug_{1 if cfg_b.class_imbalance else 0}-"
            f"vTTA_{1 if cfg_b.val_TTA else 0}-"
            f"tTTA_{1 if cfg_b.test_TTA else 0}-"
            f"MP_{1 if cfg_b.mixed_precision else 0}"
        )

        run = None 
        if hasattr(cfg_a, 'wandb') and cfg_a.wandb['log']:
            run = wandb.init(
                project=f"{cfg_a.wandb['project']}-A{cfg_a.model_name[:15]}-B{cfg_b.model_name[:15]}",
                name=next_run_name,
                config=vars(cfg_a),
            )

        ### submission 폴더 생성
        # 모델 저장, 시각화 그래프 저장, submission 파일 등등 저장 용도
        submission_dir = os.path.join(cfg_a.data_dir, 'submissions', next_run_name)
        try:
            os.makedirs(submission_dir, exist_ok=False)
            # cfg에 추가 
            cfg_a.submission_dir = submission_dir
            cfg_b.submission_dir = submission_dir
        except:
            raise ValueError("같은 이름의 submission 폴더가 있습니다.", submission_dir)


        ### Data Load
        df = pd.read_csv(os.path.join(cfg_a.data_dir, cfg_a.train_data))
        # --- 증강 설정 ---
        train_transforms_a, val_transform_a, val_tta_transform_a, test_tta_transform_a = get_augmentation(cfg_a, epoch=0)
        train_transforms_b, val_transform_b, val_tta_transform_b, test_tta_transform_b = get_augmentation(cfg_b, epoch=0)
        # --- 모델 학습 ---
        trainer_a = train_hierarchical_model('A', cfg_a, df, run, train_transforms_a, val_transform_a, val_tta_transform=val_tta_transform_a)
        
        # end Model A run
        if run is not None:
            run.finish()
        
        run = None 
        if hasattr(cfg_b, 'wandb') and cfg_b.wandb['log']:
            run = wandb.init(
                project=f"{cfg_b.wandb['project']}-A{cfg_a.model_name[:15]}-B{cfg_b.model_name[:15]}",
                name=next_run_name2,
                config=vars(cfg_b),
            )

        trainer_b = train_hierarchical_model('B', cfg_b, df, run, train_transforms_b, val_transform_b, val_tta_transform=val_tta_transform_b)

        if run is not None:
            run.finish()
            run = None

        # 추론 파트
        print("===== Starting Hierarchical Inference =====")
        test_df = pd.read_csv(os.path.join(cfg_a.data_dir, "sample_submission.csv"))
        print("Step a) Predicting with Model A...")
        test_dataset_a = ImageDataset(test_df, os.path.join(cfg_a.data_dir, "test"), transform=val_transform_a)
        test_loader_a = DataLoader(test_dataset_a, batch_size=cfg_a.batch_size, shuffle=False, num_workers=8, pin_memory=True)    
        if cfg_a.test_TTA:
            print("Running TTA on test set...")
            preds_a = tta_predict(trainer_a.model, test_dataset_a, test_tta_transform_a, device, cfg_a, flag='test')
        else:
            print("Running inference on test set...")
            preds_a = predict(trainer_a.model, test_loader_a, device)
        
        preds_a = predict(trainer_a.model, test_loader_a, device)
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
            if cfg_b.test_TTA:
                test_dataset_raw_b = ImageDataset(test_df, os.path.join(cfg_b.data_dir, "test"), transform=val_transform_b)
                # test_loader_raw_b = DataLoader(test_dataset_raw_b, batch_size=cfg_b.batch_size, shuffle=False, num_workers=8, pin_memory=True)
                print("Running TTA on test set...")
                preds_b = tta_predict(trainer_b.model, test_dataset_raw_b, test_tta_transform_b, device, cfg_b, flag='test')
            else:
                test_dataset_b = ImageDataset(ids_for_model_b, os.path.join(cfg_b.data_dir, "test"), transform=val_transform_b)
                test_loader_b = DataLoader(test_dataset_a, batch_size=cfg_b.batch_size, shuffle=False, num_workers=8, pin_memory=True)    
                print("Running inference on test set...")
                preds_b = predict(trainer_b.model, test_loader_b, device)
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
        submission_path = os.path.join(submission_dir, f"{next_run_name}.csv")
        final_submission_df.to_csv(submission_path, index=False)
        print(f"📢 Submission file saved to {submission_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if run is not None:
            run.finish()
        # Note: Offline augmented images are deleted within the training functions.