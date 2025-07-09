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

#📢 project_root 설정 필수
project_root = '/data/ephemeral/home/upstageailab-cv-classification-cv_5'
sys.path.append(project_root)
from codes.gemini_utils_v2_2 import *
from codes.gemini_train_v2_2 import *
from codes.gemini_augmentation_v2_2 import *
from codes.gemini_evalute_v2_2 import *

def run_training_cycle(train_df, val_df, cfg, run, train_transforms, val_transform):
    augmented_ids, val_augmented_ids = [], []
    # 클래스 불균형 해소를 위한 이미지 offline 증강
    if hasattr(cfg, 'class_imbalance') and cfg.class_imbalance:
        new_augmented_ids, augmented_labels = augment_class_imbalance(cfg, train_df)
        augmented_ids.extend(new_augmented_ids)
        imb_aug_df = pd.DataFrame({"ID": new_augmented_ids, "target": augmented_labels})
        train_df = pd.concat([train_df, imb_aug_df], ignore_index=True).reset_index(drop=True)

    # validation 데이터를 offline으로 eda 증강을 적용
    if val_df is not None and cfg.val_TTA:
        new_val_augmented_ids, augmented_labels = augment_validation(cfg, val_df)
        val_augmented_ids.extend(new_val_augmented_ids)
        val_aug_df = pd.DataFrame({"ID": new_val_augmented_ids, "target": augmented_labels})
        val_df = pd.concat([val_df, val_aug_df], ignore_index=True).reset_index(drop=True)

    # Sampler 설정
    sampler = None
    shuffle = True
    if cfg.weighted_random_sampler:
        targets = train_df['target'].values
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        weights = class_weights[targets]
        g = get_generator(cfg)
        sampler = WeightedRandomSampler(weights, len(weights), generator=g)
        shuffle = False

    # Dataset 생성
    if cfg.online_augmentation:
        train_dataset = ImageDataset(train_df, os.path.join(cfg.data_dir, "train"), transform=train_transforms[0])
    else:
        datasets = [ImageDataset(train_df, os.path.join(cfg.data_dir, "train"), transform=t) for t in train_transforms]
        train_dataset = ConcatDataset(datasets)
    
    val_loader = None
    if val_df is not None:
        val_dataset = ImageDataset(val_df, os.path.join(cfg.data_dir, "train"), transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if cfg.online_aug['mixup']:
        train_collate = lambda batch: mixup_collate_fn(batch, num_classes=17, alpha=0.4)
    elif cfg.online_aug['cutmix']:
        train_collate = lambda batch: cutmix_collate_fn(batch, num_classes=17, alpha=0.4)
    else:
        train_collate = None
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=shuffle, num_workers=8, pin_memory=True, collate_fn=train_collate)

    # TrainModule 정의
    model = get_timm_model(cfg)
    class_weights = None
    if hasattr(cfg, 'class_weighting') and cfg.class_weighting:
        class_counts = train_df['target'].value_counts().sort_index()
        weights = 1.0 / class_counts
        class_weights = torch.tensor(weights.values, dtype=torch.float32).to(cfg.device)
    
    criterion = get_criterion(cfg, class_weights=class_weights)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    trainer = TrainModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=val_loader,
        cfg=cfg,
        verbose=1,
        run=run
    )

    # 학습
    train_result = trainer.training_loop()
    if not train_result:
        raise ValueError("Failed to train model...")

    return trainer, augmented_ids, val_augmented_ids, val_df, val_loader

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

        args = parser.parse_args()

        # Yaml 파일 읽기
        cfg = load_config(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
        )
        # 랜덤성 제어
        set_seed(cfg.random_seed)

        # device 설정
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        print("⚙️ Device :",device)
        cfg.device = device
        CURRENT_TIME = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%y%m%d%H%M")
        print(f"⌚ 실험 시간: {CURRENT_TIME}")

        # W&B 설정
        # 증강 기법 문자열 생성 로직 개선
        aug_str_parts = ""
        if cfg.online_augmentation:
            aug_str_parts += "on"
            if cfg.dynamic_augmentation and cfg.dynamic_augmentation['enabled']:
                aug_str_parts += "daug"                
            else:
                # 일반 augmentation 기법들만 나열
                aug_str_parts += "aug"
        else:
            aug_str_parts += "offaug"

        next_run_name = (
            f"{CURRENT_TIME}-"
            f"{cfg.model_name}-"
            f"opt_{cfg.optimizer_name}-"
            f"sch_{cfg.scheduler_name}-"
            f"img{cfg.image_size}-"
            f"es{cfg.patience}-"
            f"{aug_str_parts}-"  # 개선된 증강 문자열
            f"cv{cfg.n_folds}-"
            f"clsaug_{1 if cfg.class_imbalance else 0}-"
            f"vTTA_{1 if cfg.val_TTA else 0}-"
            f"tTTA_{1 if cfg.test_TTA else 0}-"
            f"MP_{1 if cfg.mixed_precision else 0}"
        )

        run = None 
        if hasattr(cfg, 'wandb') and cfg.wandb['log']:
            run = wandb.init(
                project=cfg.wandb['project'],
                name=next_run_name,
                config=vars(cfg),
            )

        ### submission 폴더 생성
        # 모델 저장, 시각화 그래프 저장, submission 파일 등등 저장 용도
        submission_dir = os.path.join(cfg.data_dir, 'submissions', next_run_name)
        try:
            os.makedirs(submission_dir, exist_ok=False)
            # cfg에 추가 
            cfg.submission_dir = submission_dir
        except:
            raise ValueError("같은 이름의 submission 폴더가 있습니다.", submission_dir)


        ### Data Load
        df = pd.read_csv(os.path.join(cfg.data_dir, cfg.train_data))
        # Augmentation 설정
        train_transforms, val_transform, val_tta_transform, test_tta_transform = get_augmentation(cfg, epoch=0)
        val_augmented_ids = []
        
        # Cross validation if n_folds >= 3
        if cfg.n_folds >= 3:
            # Augmentation 설정    
            # train_transforms, val_transform, val_tta_transform, test_tta_transform = get_augmentation(cfg, epoch=0)
            
            folds_es, folds_val_f1 = [], []

            skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed)
            for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
                try:
                    train_losses_for_plot, val_losses_for_plot = [], []
                    train_acc_for_plot, val_acc_for_plot = [], []
                    train_f1_for_plot, val_f1_for_plot = [], []

                    print(f"===== FOLD {fold+1} =====")
                    print("="*20)
                    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

                    trainer, fold_augmented_ids, fold_val_augmented_ids, val_df, val_loader = run_training_cycle(
                        train_df, val_df, cfg, run=None, train_transforms=train_transforms, val_transform=val_transform
                    ) # cross validation 시에는 wandb에 기록하지 않는다.
                    # 명시적으로 best model state dict 복원
                    trainer.es.restore_best(trainer.model)

                    # save fold results
                    train_losses_for_plot.append(trainer.train_losses_for_plot)
                    train_acc_for_plot.append(trainer.train_acc_for_plot)
                    train_f1_for_plot.append(trainer.train_f1_for_plot)
                    val_losses_for_plot.append(trainer.val_losses_for_plot)
                    val_acc_for_plot.append(trainer.val_acc_for_plot)
                    val_f1_for_plot.append(trainer.val_f1_for_plot)

                    # fold early stopped moment
                    folds_es.append(trainer.es.best_loss_epoch)

                    # evaluate
                    val_preds, val_f1 = do_validation(
                        df=val_df,
                        model=trainer.model,
                        data=val_loader,
                        transform_func=val_tta_transform,
                        cfg=cfg,
                        run=None,
                        show=False,
                        savepath=os.path.join(cfg.submission_dir, f"val_confusion_matrix{'_TTA' if cfg.val_TTA else ''}_Fold{fold}.png")
                    )
                    folds_val_f1.append(val_f1)
                finally:
                    delete_offline_augmented_images(cfg=cfg, augmented_ids=fold_augmented_ids)
                    delete_offline_augmented_images(cfg=cfg, augmented_ids=fold_val_augmented_ids)
                
                print("="*20)
                print("="*20)

            # out of cross-validation
            # 1. plot
            # 2. print average val f1
            # 3. set epoch
            # 4. train whole dataset & make final model
            plot_cross_validation(train_losses_for_plot, val_losses_for_plot, "Loss", cfg, show=False)
            plot_cross_validation(train_acc_for_plot, val_acc_for_plot, "Accuracy", cfg, show=False)
            plot_cross_validation(train_f1_for_plot, val_f1_for_plot, "F1-score", cfg, show=False)
            best_epoch = int(np.mean(folds_es))
            print(f"📢  Avg F1: {np.mean(folds_val_f1):.5f}, Best Epoch: {best_epoch}")
            # 전체 학습.
            trainer, _, _, _ = run_training_cycle(
                df, None, cfg, run, train_transforms=train_transforms, val_transform=val_transform
            ) # 전체 train 데이터를 사용해 학습. val_df 없음.
            # 명시적으로 best model state dict 복원
            trainer.es.restore_best(trainer.model)
            trainer.save_experiments(savepath=os.path.join(cfg.submission_dir, f'{next_run_name}.pth'))
            

        # No Cross Validation
        else:
            try:
                # Train-validation 분할
                train_df, val_df = train_test_split(df, test_size=cfg.val_split_ratio, random_state=cfg.random_seed, stratify=df['target'] if cfg.stratify else None)
                
                trainer, augmented_ids, val_augmented_ids, val_df, val_loader = run_training_cycle(
                    train_df, val_df, cfg, run, train_transforms=train_transforms, val_transform=val_transform
                )
                # 명시적으로 best model state dict 복원
                trainer.es.restore_best(trainer.model)

                ### Save Model
                trainer.save_experiments(savepath=os.path.join(cfg.submission_dir, f'{next_run_name}.pth'))
                ## 학습 결과 시각화 저장.
                trainer.plot_loss(
                    show=False,
                    savewandb=cfg.wandb['log'],
                    savedir=cfg.submission_dir
                )

                ### Evaluate
                val_preds, val_f1 = do_validation(
                    df=val_df, 
                    model=trainer.model, 
                    # data=val_dataset_raw if cfg.val_TTA else val_loader, # online validation TTA
                    data = val_loader, # offline validation TTA
                    transform_func=val_tta_transform, 
                    cfg=cfg, 
                    run=run, 
                    show=False, 
                    savepath=os.path.join(cfg.submission_dir, f"val_confusion_matrix{'_TTA' if cfg.val_TTA else ''}.png")
                )
                print("📢 Validation F1-score:",val_f1)
            finally:
                delete_offline_augmented_images(cfg=cfg, augmented_ids=augmented_ids)
                delete_offline_augmented_images(cfg=cfg, augmented_ids=val_augmented_ids)

            # Save incorrect validation results
            try:
                save_validation_images(val_df, val_preds, cfg, images_per_row=5, show=False)
            except:
                print("⚠️Saving incorrect validation results Failed...")

        # Inference
        test_df = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))

        test_dataset = ImageDataset(test_df, os.path.join(cfg.data_dir, "test"), transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        tta_augs = [
            A.Compose([A.HorizontalFlip(p=1.0), test_tta_transform]), # 수평 반전
            A.Compose([A.VerticalFlip(p=1.0), test_tta_transform]),   # 수직 반전
            A.Compose([A.Transpose(p=1.0), test_tta_transform]),      # 대칭 (Transposition)
            A.Compose([A.Rotate(limit=(-10, 10), p=1.0), test_tta_transform]), # 미세한 회전
            test_tta_transform # 증강하지 않는 원본 이미지 변환 (마지막에 추가)
        ]
        test_tta_dataset = TestTTAImageDataset(
            dataframe=test_df,
            img_dir=os.path.join(cfg.data_dir,"test"),
            transforms_list=tta_augs
        )

        if cfg.test_TTA:
            print("Running TTA on test set...")
            test_preds = tta_predict(trainer.model, test_tta_dataset, tta_augs, device, cfg, flag='test')
        else:
            print("Running inference on test set...")
            test_preds = predict(trainer.model, test_loader, device)

        print(pd.Series(test_preds).value_counts())
        pred_df = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))
        pred_df['target'] = test_preds

        # Submission
        sample_submission_df = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))
        assert (sample_submission_df['ID'] == pred_df['ID']).all(), "pred_df에서 test 이미지가 아닌 데이터가 존재합니다."
        assert set(pred_df['target']).issubset(set(range(17))), "target 컬럼에 0~16 외의 값이 있습니다."

        submission_path = os.path.join(cfg.submission_dir, f"{next_run_name}.csv")
        pred_df.to_csv(submission_path, index=False)
        print(f"📢Submission file saved to {submission_path}")

        ### prediction class별 개수
        try:
            class_counts = pred_df['target'].value_counts().sort_index()
            class_counts = class_counts.reset_index(drop=False)
            meta = pd.read_csv(os.path.join(cfg.data_dir, "meta.csv"))
            meta_dict = zip(meta['target'], meta['class_name'])
            meta_dict = dict(meta_dict)
            targets_class = list(map(lambda x: meta_dict[x], class_counts['target']))
            class_counts['meta'] = targets_class
            class_counts.to_csv(os.path.join(cfg.submission_dir, "submission_class_distribution.csv"), index=False)
        except Exception as e:
            print(e)

        if run:
            # Log submission artifact
            artifact = wandb.Artifact(f'submission-{next_run_name[:60]}...', type='submission')
            artifact.add_file(submission_path)
            run.log_artifact(artifact)
            run.finish()

    finally:
        if run:
            run.finish()
        if augmented_ids:
            ### Offline Augmentation 파일 삭제
            delete_offline_augmented_images(cfg=cfg, augmented_ids=augmented_ids)
        if val_augmented_ids:
            ### Offline Augmentation 파일 삭제
            delete_offline_augmented_images(cfg=cfg, augmented_ids=val_augmented_ids)