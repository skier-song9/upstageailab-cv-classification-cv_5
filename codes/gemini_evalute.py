import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import wandb

def tta_predict(model, dataset, tta_transform, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for enumidx, (image, _) in enumerate(tqdm(dataset, desc="TTA Prediction")): # load batch
            # read raw images from dataset_raw
            tta_preds = []
            image = image.clamp(0, 255).to(torch.uint8) 
            image = image.permute(1, 2, 0).cpu().numpy() # H,W,C 로 변형
            for _ in range(5): # 5 TTA iterations
                augmented_image = tta_transform(image=image)['image']
                augmented_image = augmented_image.to(device)
                augmented_image = augmented_image.unsqueeze(0) # batch, H,W,C 로 변형
                outputs = model(augmented_image) # inference
                tta_preds.append(outputs.softmax(1).cpu().numpy()) # append inference result
        
            avg_preds = np.mean(tta_preds, axis=0) # 5 TTA 예측 결과 확률값을 평균 낸다.
            predictions.extend(avg_preds.argmax(1))
    return predictions

def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Prediction"):
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.argmax(1).cpu().numpy())
    return predictions

def do_validation(df, model, data, transform_func, cfg, run=None, show=False, savepath=None):
    if cfg.TTA:
        print("Running TTA on validation set...")
        val_preds = tta_predict(model, data, transform_func, cfg.device)
    else:
        print("Running Normal Validation...")
        val_preds = predict(model, data, cfg.device)
    val_targets = df['target'].values
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    # 메타데이터 로드
    meta = pd.read_csv(os.path.join(cfg.data_dir, 'meta.csv'))
    meta_dict = zip(meta['target'], meta['class_name'])
    meta_dict = dict(meta_dict)
    # meta_dict[-1] = "None"
    val_targets_class = list(map(lambda x: meta_dict[x], val_targets))
    val_preds_class = list(map(lambda x: meta_dict[x], val_preds))
    all_classes = sorted(list(set(val_targets_class + val_preds_class)))
    cm = confusion_matrix(val_targets_class, val_preds_class, labels=all_classes)
    plt.figure(figsize=(10, 8), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_classes, yticklabels=all_classes)
    plt.title(f"Validation Confusion Matrix - F1: {val_f1:.4f}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if run:
        run.log({"tta_val_confusion_matrix": wandb.Image(plt)})
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.clf()
    return val_preds, val_f1
