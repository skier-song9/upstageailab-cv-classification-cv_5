import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


def get_augmentation(cfg):
    common_resize_transform = A.Compose([
        # 긴 변을 기준으로 종횡비를 유지하며 resize
        A.LongestMaxSize(max_size=cfg.image_size),
        # cfg.image_size 정사각형으로 만들고, 여백은 흰색으로 채움.
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=1.0),
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ])

    AUG = {
        'eda': A.Compose([
            # 공간 변형에 대한 증강
            A.ShiftScaleRotate(shift_limit=(-0.05,0.05), scale_limit=(-0.15, 0.15), rotate_limit=(-20, 30), fill=(255,255,255), p=0.9),
            # x,y 좌표 반전 
            A.Transpose(p=0.5),
            # Blur & Noise
            A.OneOf([
                A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                A.Blur(blur_limit=(3, 9), p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.0025, 0.2), p=0.8),
            # Brightness, Contrast, ColorJitter
            A.ColorJitter(brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=0.8),
        ]),
        'dilation': A.Compose([
            A.Morphological(p=1, scale=(1, 3), operation="dilation"),
            # 공간 변형에 대한 증강
            A.ShiftScaleRotate(shift_limit=(-0.05,0.05), scale_limit=(-0.15, 0.15), rotate_limit=(-20, 30), fill=(255,255,255), p=0.9),
            # x,y 좌표 반전 
            A.Transpose(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
            A.RandomBrightnessContrast(p=1),
        ]),
        'erosion': A.Compose([
            A.Morphological(p=1, scale=(2, 4), operation="erosion"),
            # 공간 변형에 대한 증강
            A.ShiftScaleRotate(shift_limit=(-0.05,0.05), scale_limit=(-0.15, 0.15), rotate_limit=(-20, 30), fill=(255,255,255), p=0.9),
            # x,y 좌표 반전 
            A.Transpose(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
            A.RandomBrightnessContrast(p=1),
        ]),
        
    }

    train_transforms = []
    active_augs = [AUG[aug] for aug, active in cfg.augmentation.items() if active and aug in AUG]

    if cfg.online_augmentation:
        # online augmentation 학습 : 실시간으로 증강 기법을 적용하여, 더 다양한 증강 형태의 데이터를 학습할 수 있다.
        # 장점 : 무한한 다양성, 과적합 방지 효과 증대, 저장 공간 효율성
        # 단점 : 전처리 과정의 증가로 학습 시간 증가, 재현성이 떨어짐. 너무 많은 증강 기법을 적용하면, 의도치 않은 결과가 나올 수 있다.
        if active_augs:
            online_transform = A.Compose([
                A.OneOf(active_augs, p=0.85), # 85% 확률로 active_augs에 설정된 증강 기법들이 적용된다. 15% 확률로 원본 train 데이터를 사용한다.
                common_resize_transform # Resize 기법은 항상 동일하게.
            ])
            train_transforms.append(online_transform)
        else: # 따로 지정한 증강 기법이 없는 경우, Resize 기법만 사용
            train_transforms.append(common_resize_transform)
    else:
        # offline augmentation 학습 : 개별적으로 dataset을 만들어 ConcatDataset을 최종 생성한다. 모든 증강 기법을 적용 가능하고, 마치 데이터셋 개수 자체가 늘어난 것 같은 효과를 준다. (원래라면, 증강한 데이터를 저장해야 하지만 이건 생략.)
        # 단점 : 다양성 제한
        if active_augs:
            for aug_pipeline in active_augs: # 각각의 aug를 transform_func으로 만든다.
                train_transforms.append(A.Compose([aug_pipeline, common_resize_transform]))
        else:
            train_transforms.append(common_resize_transform)

    # validation 증강은 기본 증강만 사용한다.
    val_transform = common_resize_transform

    # Validation transform with 'eda' augmentation to simulate test conditions
    # tta_transform은 TTA에서도 사용할 증강이다.
    tta_transform = A.Compose([
        AUG['eda'],
        common_resize_transform
    ])

    return train_transforms, val_transform, tta_transform

### Offline augmentation
def augment_class_imbalance(cfg, train_df):
    # Cutout 증강 파이프라인 설정
    cutout_transform = A.Compose([
        # 이미지 크기 조정
        A.CoarseDropout(
            num_holes_range=(1, 2), # 마스킹 개수
            hole_height_range=(int(cfg.image_size * 0.05), int(cfg.image_size * 0.1)), # 마스킹의 높이 범위
            hole_width_range=(int(cfg.image_size * 0.05), int(cfg.image_size * 0.2)), # 마스킹의 너비 범위
            fill=(0,0,0), # 검정색 마스킹
            p=1.0
        )
    ])
    # 증강 대상 클래스
    augment_classes = cfg.class_imbalance['aug_class']
    max_samples = cfg.class_imbalance['max_samples']

    # 증강 이미지, 라벨, ID 리스트 초기화
    augmented_labels = []    # 증강된 이미지 라벨
    augmented_ids = []       # 증강된 이미지 ID
    total_augmented = 0
    # 증강 대상 클래스 루프
    for cls in augment_classes:
        print(cls, "클래스")
        cls_df = train_df[train_df['target'] == cls]
        current_count = len(cls_df)
        print("현재 개수:", current_count)
        # 목표 샘플 수에 도달하기 위해 필요한 증강 이미지 개수 계산
        # (current_count가 max_samples보다 많으면 0 또는 음수가 됨)
        to_generate = max_samples - current_count
        print("증강 개수:", to_generate)
        # 만약 현재 이미지 수가 목표치보다 많거나 같으면 증강할 필요 없으므로 다음 클래스로 넘어감
        if to_generate <= 0:
            continue
        # 증강할 이미지들을 원본 데이터프레임에서 샘플링 (중복 허용: replace=True)
        # 목표 샘플 수 (to_generate)만큼 이미지를 샘플링하며,
        # 만약 현재 이미지 개수(current_count)가 to_generate보다 적으면 중복 선택(replace=True)
        # 이 부분은 원래 코드의 `if/else` 조건문을 통합한 것입니다.
        sampled_df = cls_df.sample(n=to_generate, replace=False, random_state=cfg.random_seed).reset_index(drop=True)
        # 샘플링된 각 이미지에 대해 증강 수행 및 저장
        for idx, row in sampled_df.iterrows():
            img_id = row['ID']
            img_path = os.path.join(cfg.data_dir, 'train', img_id)
            # 이미지 로드 (OpenCV는 기본적으로 BGR로 로드)
            img = cv2.imread(img_path)
            # BGR 이미지를 RGB로 변환 (Albumentations는 RGB를 기대)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 정의된 Cutout 증강 적용
            augmented_img = cutout_transform(image=img)['image']

            # 증강된 이미지의 새로운 ID 생성 (기존 ID 앞에 'aug_' 접두사 추가)
            new_id = f"aug_{img_id}"
            save_path = os.path.join(cfg.data_dir, 'train', new_id)

            # 증강된 RGB 이미지를 다시 BGR로 변환하여 파일로 저장
            cv2.imwrite(save_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

            # 증강된 이미지의 라벨과 ID를 리스트에 추가
            augmented_labels.append(cls)
            augmented_ids.append(new_id)
            total_augmented += 1
        print(f"총 {total_augmented} 개의 이미지 증강")
    return augmented_ids, augmented_labels
def delete_offline_augmented_images(cfg, augmented_ids):
    train_dir = os.path.join(cfg.data_dir, 'train')
    _ = 0
    for filename in augmented_ids:
        file_path = os.path.join(train_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            _ += 1
        else:
            print("Wrong filename:", file_path)
    print(_,"개 이미지 제거")
