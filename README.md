# 문서 이미지 분류 경진대회: 고급 증강 및 어텐션 전략

## Team

| ![송규헌](https://github.com/user-attachments/assets/dfe882c4-1a19-4487-a165-b31490002750) | ![이상현](https://github.com/user-attachments/assets/d292ca56-e0cb-4f7f-8e8f-42f581464139) | ![이영준](https://github.com/user-attachments/assets/e3fa8539-bcbd-462f-bbd1-641f16f22ebc) | ![조은별](https://github.com/user-attachments/assets/b755e973-4d09-43af-8c6d-f95b17e0b066) | ![편아현](https://github.com/user-attachments/assets/991dc1aa-04f7-45d7-9189-4316e7f5c62b) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [송규헌](https://github.com/skier-song9)             |            [이상현](https://github.com/yourshlee)             |            [이영준](https://github.com/ProDevJune)             |            [조은별](https://github.com/UpstageAILab)             |            [편아현](https://github.com/vusdkvus1)             |
|                            팀장                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |
|   프로젝트 전체 기획 및 실험 리딩, 주요 EDA 설계, 베이스라인 모델 구현, 2-Stage 모델링 전략 수립 및 실험 전개, 실험 방향 설정 및 팀 내 기술 조율      |     데이터 특성 분석 및 문제 정의, EDA 기여, 클래스 불균형 대응 전략 수립, Head 특화 모델 실험 수행        |   실험 자동화 환경 구축, 다양한 모델 및 하이퍼파라미터 조합 실험 반복 수행    |    마스킹 탐지 및 통계 기반 분석, MaxViT 모델 실험 및 파라미터 최적화, 스케줄러 실험  |  MixUp 구현, 오프라인 증강 전략 개발, 증강 기법 조합 실험, ConvNeXt 모델 실험    |

## 0. Overview
### Environment
- **OS**: linux
- **Python**: 3.10+
- **PyTorch**: 2.5.0 (CUDA 12.1)
- ** 주요 라이브러리**: `timm`, `albumentations`, `pandas`, `numpy`, `scikit-learn`, `wandb`, 

### Requirements
- 필요한 라이브러리는 `uv` 를 통해 설치할 수 있습니다.
> guide : [Getting Started⭐](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/issues/5)
```bash
uv pip install -r requirements.txt
uv pip install -r pyproject.toml
```

## 1. Competiton Info

### Overview
- **대회명**: 문서 이미지 유형 분류 (Document Image Type Classification)
- **목표**: 주어진 문서 이미지를 17개의 클래스로 분류하는 Task
- **평가 지표**: Macro F1-Score

### Timeline
- 

## 2. Components

### Directory
```
├── codes/
│   ├── Baseline_augment.ipynb
│   ├── gemini_main.py
│   ├── gemini_main_v2.py
│   ├── gemini_main_v2_2.py
│   └── gemini_custom_header.py  # 최종 모델 및 훈련 로직
├── configs/
│   ├── config_v2.py 		# main_v2.py 설정 파일
│   ├── config_v2_1.py		# main_v2_1.py 설정 파일
│   └── config_v2_2.py		# main_v2_2.py 설정 파일
├── data/
│   ├── train/
│   ├── test/
│   └── train.csv
├── docs/
│   ├── Dataset EDA.md
│   └── 데이터 증강 전략.md
├── models/                      # 훈련된 모델 가중치 저장
└── README.md
```

## 3. Data descrption

### Dataset overview
- **Train**
  - 총 **1,570장**의 문서 이미지로 구성
  - 각 이미지에 대해 **17개 클래스 중 하나**로 라벨링되어 있음
- **Test**
  - 총 **3,140장**의 문서 이미지
  - 라벨은 제공되지 않으며, 최종 예측 제출용
- **Class 종류 (총 17종)**
  - 예시: `진단서`, `여권`, `차량등록증`, `신분증` 등 다양한 문서 포함
- **특징**
  - 클래스 간 **데이터 불균형** 존재
  - **Test 데이터는 Train보다 더 많은 왜곡** (회전, 노이즈, 블러 등) 포함
	
### 학습 데이터 구성

#### `train/`
- 1,570장의 이미지 파일(`.png` 등) 저장됨

#### 📄 `train.csv`
- 각 이미지의 **ID**와 **클래스 라벨 번호** 제공
- 총 1,570행
![train](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/832b4982-bd93-4480-936f-3c93a1aee98b.png)

#### 📄 `meta.csv`
- 클래스 번호(`target`)와 해당 이름(`class_name`) 매핑 정보
- 총 17행
![meta](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/d4b872ca-b669-4166-b146-5ce12af01deb.png)


### 평가 데이터 구성

#### `test/`
- 3,140장의 이미지 파일 저장됨 (라벨 없음)

#### 📄 `sample_submission.csv`
- 제출용 샘플 파일
- 총 3,140행 (Test 이미지와 동일 수)
- `target` 값은 전부 0으로 채워져 있음 (예측값 입력 필요)

![sample_submission](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/86c6b7ed-f8a4-4909-a614-a8d3bdfc94a7.png)

- 그 밖에 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 되었고 훼손된 이미지들이 존재

### EDA
`docs/Dataset EDA.md` 문서에 상세한 분석 과정이 기록되어 있습니다. 주요 발견 사항은 다음과 같습니다.

#### **밝기 및 색상**: Test 데이터셋이 Train 데이터셋보다 전반적으로 더 밝고(brightness), RGB 채널 분포가 오른쪽으로 치우쳐 있습니다.
- 밝기 (Brightness): 이미지를 흑백으로 변환하여 픽셀의 중앙값 분포를 파악하여 밝기 분석
- 대비 (Contrast): 이미지를 흑백으로 변환하여 픽셀 값의 표준편차를 파악
| Brightness_Train | Brightness_Test |
|:----------------:|:----------------:|
|![Brightness_train](https://private-user-images.githubusercontent.com/113088511/460978521-a5f38a1d-a0ea-4b3f-9164-a0be1f463ec9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI1OTIsIm5iZiI6MTc1MjIwMjI5MiwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4NTIxLWE1ZjM4YTFkLWEwZWEtNGIzZi05MTY0LWEwYmUxZjQ2M2VjOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjUxMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lYTY5OWM1OGI4ODk5MDNiNWFhYzZiZmExYjgzYmNiNDk2MGRiMDZlNzVlODQ1NzdiMTY2NmViYjk0YTQ0YjQ1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ZEbF4jEkah30kivuyUOtoPNg5WYGGOUt7ZpNFmQQDGY)|
![Brightness_test](https://private-user-images.githubusercontent.com/113088511/460978563-cf5ce0ad-0c5c-470e-b5d9-b628e8053d1e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI1OTIsIm5iZiI6MTc1MjIwMjI5MiwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4NTYzLWNmNWNlMGFkLTBjNWMtNDcwZS1iNWQ5LWI2MjhlODA1M2QxZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjUxMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03OWQ4YTQxMjgyY2M2ZTcxZWRmNjlkMjJiZTdkYzdlZGFkN2YxYTQ1ZmM1YTYzM2JhNGVjMDQzNTRjMDg2ZDAxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.xxmvh2fOOVqpprrlpsNinItC4AowrNM154-3sw_FmYI)|

| Contrast_Train | Contrast_Test |
|:--------------:|:-------------:|
![Contrast_train](https://private-user-images.githubusercontent.com/113088511/460978452-df6a7166-e5c4-4a7e-b9cc-c06c541604e0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI5OTcsIm5iZiI6MTc1MjIwMjY5NywicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4NDUyLWRmNmE3MTY2LWU1YzQtNGE3ZS1iOWNjLWMwNmM1NDE2MDRlMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjU4MTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iYzZkZTNmMDU1N2IyNjU2MjgxZDY0ZmRkN2IxMTkxZGE5NDMwYWI4NGE0NTY2NWRjYzAzNTQzZGI0YTIzMGE2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.E7Wg2pNExtxOpMNenT6njAclhxAptIBAOa7dgTVFHjQ)|
![Contrast_test](https://private-user-images.githubusercontent.com/113088511/460978491-f9cd86c8-c08a-4945-9d94-f2ca9efa4def.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI5OTcsIm5iZiI6MTc1MjIwMjY5NywicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4NDkxLWY5Y2Q4NmM4LWMwOGEtNDk0NS05ZDk0LWYyY2E5ZWZhNGRlZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjU4MTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04ZDEzM2YyZTFlNTY5ZTI4OWE0MGY0ZmQxZWZhYzkzNzYyYzI4ZTBjZDAzMDU4ZDM5Mzc0NmQ2NjhlZGQzMTg0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.lncYcFrmMB6lvSKRM2IUTOTk1uAqt99XnX1W_IuELjo)|


#### **선명도 및 노이즈**: Test 데이터셋은 Train 데이터셋보다 더 흐릿하고(blur), 노이즈가 많습니다.
- 흐림 정도 (Blurriness): 이미지를 흑백으로 변환하여 Laplacian 필터를 적용한 후, 분산으로 blur 정도를 분석
- 노이즈 추정 (Noise Estimation): 웨이블렛 변환을 사용하여 이미지를 다양한 주파수 대역으로 분해, 이미지의 노이즈 표준 편차를 추정
| Brightness_Train | Brightness_Test |
|:----------------:|:----------------:|
|![Blurriness_train](https://private-user-images.githubusercontent.com/113088511/460978339-c22358eb-d076-44f3-a4dc-9c74d0fbad97.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDIzMDQsIm5iZiI6MTc1MjIwMjAwNCwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4MzM5LWMyMjM1OGViLWQwNzYtNDRmMy1hNGRjLTljNzRkMGZiYWQ5Ny5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjQ2NDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01YjI1YTM3ODNhZDQxNGFmYmZhNWRiYzE4YjRhM2ZkYzVlMjljNTYxZTRkMTg3NWExYjI2ZDM4NTA4MTNlZGQ4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.3NzFCLNdiR-6V8tuAEhwPs6ILnyyMHYGjA02TqhVpuk)|
![Blurriness_test](https://private-user-images.githubusercontent.com/113088511/460978394-b87d1fc9-06cf-46ac-b3b3-0641fe82c3a7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDIzMDQsIm5iZiI6MTc1MjIwMjAwNCwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4Mzk0LWI4N2QxZmM5LTA2Y2YtNDZhYy1iM2IzLTA2NDFmZTgyYzNhNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjQ2NDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hZWY5MDQ5MDA2MzQ2MWE3MTkzNzE5NTQ1MzA1ZmFjNWJiNGM2ZTNiNDI3MDBiZjZhMmMxNDdjZmZiYjk2YTVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.2r3WnQ61hm7bgM7BCF70Gk8Ekpy3MFpACNgTJUaEuuM)

| Noise_Train | Noise_Test |
|:----------------:|:----------------:|
|![Noise_train](https://private-user-images.githubusercontent.com/113088511/460978146-7276c8d7-8288-4e6b-93d1-30a2837c50bb.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI1OTIsIm5iZiI6MTc1MjIwMjI5MiwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4MTQ2LTcyNzZjOGQ3LTgyODgtNGU2Yi05M2QxLTMwYTI4MzdjNTBiYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjUxMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05Y2M5ZGFiODY3ZjIxODZhMTRlYTZlYzM1ZTU3MzRjM2QxOTkxZjg0NzBlZWU4YjI0Y2IzZjNkZjkzYjgwODUzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.vjwVwp98p4xLKLsVQAgV8kv-dKe7eypjXKTsOZxGfZo)|
![Noise_test](https://private-user-images.githubusercontent.com/113088511/460978191-c3bb2bf8-d61f-4e4e-9886-e7cb2410f310.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDI1OTIsIm5iZiI6MTc1MjIwMjI5MiwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4MTkxLWMzYmIyYmY4LWQ2MWYtNGU0ZS05ODg2LWU3Y2IyNDEwZjMxMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMjUxMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03OTc2ZmExZWJjZjFhNWY0NGM1NzFmZWU2MzYxZWVjZjk2ZjBjZDNhMmU0Y2QxMTAwMDUxYWZkZTc2NmY4YmY4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.A5U0HDAKLWly36i02fj4xMSmZui7zUAUPaXnt83M1pI)|

#### **회전 및 종횡비**: Test 데이터셋은 0~360도의 다양한 회전이 적용된 이미지가 많으며, 특정 종횡비(4:3)의 비율이 Train 데이터셋보다 높습니다.
- 가로/세로 종횡비 (Aspect): 이미지의 너비/높이 비율 분포 분석
- 회전 추정 (Rotation Estimation): Hough Transform을 사용해 이미지에서 직선을 검출하고, 직선들의 기울기를 통해 이미지의 회적 각도를 추정
| Aspect_Train | Aspect_Test |
|:----------------:|:----------------:|
|![Aspect_train](https://private-user-images.githubusercontent.com/113088511/460978017-82ce5699-ec59-478a-acb9-90bcbbebfad3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDM1MjksIm5iZiI6MTc1MjIwMzIyOSwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4MDE3LTgyY2U1Njk5LWVjNTktNDc4YS1hY2I5LTkwYmNiYmViZmFkMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMzA3MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hOTg0MmFjNjlhMmM2MTI2Y2Y3ODYxODc1Mzc5ZWNlNGM2ZmM3MmM5YjBlYjhkOTQ5NmY5ZWYzNTYxNTE2YmViJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.FSUMaiVXSeYyhTF8thlJi8HHj0s1byieT2EoQF6FLl4)|
![Aspect_test](https://private-user-images.githubusercontent.com/113088511/460978075-34d2386e-a0b1-44c0-996b-ff4a5f787cb4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDM1MjksIm5iZiI6MTc1MjIwMzIyOSwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc4MDc1LTM0ZDIzODZlLWEwYjEtNDRjMC05OTZiLWZmNGE1Zjc4N2NiNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMzA3MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04N2IwNzdiNGEzMDA3MzI5OTliY2YwMDQ3NzBjY2JlZTk3N2UyYzM4ZWQzMjA3MTY0MzY4YWYyMDJjODk2ZjBkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.VMnQ_hMTXH5y6tKPYWw1u-b3EkYjyAIF2swHewkkcZI)|

| Rotation_Train | Rotation_Test |
|:----------------:|:----------------:|
![Rotation_train](https://private-user-images.githubusercontent.com/113088511/460977605-7bf98908-921b-40dd-b2ae-a185f6ba09f8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDM1MjksIm5iZiI6MTc1MjIwMzIyOSwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc3NjA1LTdiZjk4OTA4LTkyMWItNDBkZC1iMmFlLWExODVmNmJhMDlmOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMzA3MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zYWMzZThlODBhODBiMzc0MjIzNzc1NmQwNzUzMjE5OTIzMWNlYzNjYzA3YWQ5YzQ2OWZmMjAyYjI2MDg2NjVkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.8BSRi4nObF102uvPouTiJUmfKOT1sTvUGF3dmmeEM3o)|
![Rotation_test](https://private-user-images.githubusercontent.com/113088511/460977907-b80e41e5-44a1-4c86-a2c9-64e3909b8747.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIyMDM1MjksIm5iZiI6MTc1MjIwMzIyOSwicGF0aCI6Ii8xMTMwODg1MTEvNDYwOTc3OTA3LWI4MGU0MWU1LTQ0YTEtNGM4Ni1hMmM5LTY0ZTM5MDliODc0Ny5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcxMVQwMzA3MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zMGY1Y2EyZmQ4YTkzOTlkNmI3ZDZmNDkyZGZhNTVhOWJjNTg5OTliYmU1ZDY3ZWZlMDJkZTYwM2I3YzdjYTdkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ibcwXxxd54zgztxh9JZ91Wy3tcRWoDxXoTmm6o_mANM)|

이러한 EDA 결과는 데이터 증강 및 모델 구조 설계에 핵심적인 단서로 활용되었습니다.

### Data Processing
EDA 결과를 바탕으로 `Albumentations` 라이브러리를 사용하여 현실적인 왜곡을 시뮬레이션하는 데이터 증강 파이프라인을 구축했습니다.

- **주요 증강 기법**:
    - **기하학적 변형**: `Affine` (scale, translate, rotate, shear), `HorizontalFlip`, `VerticalFlip`, `Transpose` 등을 사용하여 회전 및 변형에 강건한 모델을 학습시켰습니다.
    - **색상 변형**: `ColorJitter`를 사용하여 Train/Test 데이터셋 간의 밝기 및 색상 분포 차이를 완화했습니다.
    - **품질 저하**: `GaussianBlur`, `GaussNoise` 등을 적용하여 Test 데이터셋의 흐림 및 노이즈 특성을 모방했습니다.
    - **문서 특화 증강**: `CoarseDropout`을 이용해 문서의 특정 영역(개인정보 등)을 마스킹하는 효과를 시뮬레이션하고, `Morphological` 연산(dilation, erosion)으로 텍스트 패턴을 강화했습니다.
- **클래스 불균형 처리**: `WeightedRandomSampler`를 사용하여 소수 클래스의 데이터가 더 자주 샘플링되도록 하여 데이터 불균형 문제를 완화했습니다.

#### Public 
- 다중 Online 증강 : 여러 증강들을 A.OneOf 로 묶어 매 epoch마다 서로 다른 증강 기법을 사용해 Online 증강을 수행한다. 
- Dynamic Online 증강 : epoch에 따라 서로 다른 증강 기법을 사용해 Online 증강을 수행한다. 
	- 처음에는 약한 강도의 증강을 적용하다가 점차 강한 강도의 증강으로 변화시킨다.
- Offline 증강 : train 데이터셋을 미리 설정한 증강 기법들로 각 클래스별로 1000개까지 증강하여 활용한다.
	- 클래스 불균형 문제가 자연스럽게 해소된다.
#### Final
- Offline 증강 + Mixup Online 증강
	- Offline 증강을 통해 클래스 불균형을 해고
- Mixup 증강을 통해 결정 경계 강화하여 어려운 클래스에 대한 예측 성능을 높임.
![aug](https://github.com/user-attachments/assets/ba94d2d2-66fb-48e6-b4a0-a100db7f29ef)

#### Grad-CAM] 
![grad_cam](https://github.com/user-attachments/assets/e3bdcaf3-789e-4860-a857-ee1a5f785bc1)
- 정보가 마스킹된 부분에 모델이 과도하게 의존할지 모른다는 우려와 달리, 훈련된 모델이 입력 이미지의 형태 자체를 기반으로 분류를 수행한다는 깨달았다.

## 4. Modeling

### Model descrition
- **Backbone**: `timm` 라이브러리의 사전 학습된 `convnext_base` 모델을 백본으로 사용했습니다. ConvNeXt는 CNN의 장점과 Transformer의 구조적 이점을 결합하여 이미지 분류에서 높은 성능을 보입니다.
- **Custom Head**: 백본 위에 커스텀 헤더를 추가하여 모델의 성능을 개선했습니다.
    - **Standard 10-View Attention**: Test 데이터셋의 다양한 회전 및 종횡비에 대응하기 위해, 이미지의 중앙, 네 코너, 그리고 각각을 수평 반전한 총 10개의 뷰(view)에서 특징을 추출하고 어텐션을 적용하여 최종 특징 벡터를 생성하는 `Standard10ViewAttention` 모듈을 구현했습니다. 이는 모델이 이미지의 여러 부분을 종합적으로 보도록 유도하여 강건성을 높입니다.
    - **Classifier**: 간단한 MLP 구조의 분류기를 사용하여 최종 클래스를 예측합니다.

### Modeling Process
모델 개발은 여러 버전에 걸쳐 점진적으로 이루어졌습니다.

- **v1 (`gemini_main.py`)**:
    - **전략**: `timm` 모델을 활용한 기본 파이프라인 구축. `Albumentations`를 사용한 기본적인 온라인/오프라인 증강 실험.
    - **주요 기술**: `TimmWrapper`, `AdamW` 옵티마이저, `StepLR` 스케줄러.

- **v2 (`gemini_main_v2.py`, `gemini_main_v2_2.py`)**:
    - **전략**: 클래스 불균형 문제 해결 및 고급 훈련 기법 도입.
    - **주요 개선 사항**:
        - **클래스 불균형**: `WeightedRandomSampler` 및 `CrossEntropyLoss`의 `class_weights` 옵션 적용.
        - **고급 증강**: `Mixup`, `Cutmix`와 같은 강력한 정규화 기법 도입.
        - **훈련 안정화**: `CosineAnnealingLR` 스케줄러 사용.
        - **신뢰도 향상**: `StratifiedKFold`를 사용한 교차 검증 도입.

- **v3 (`gemini_custom_header.py`)**:
    - **전략**: EDA 기반의 최종 모델 구조 및 훈련 전략 확립.
    - **주요 개선 사항**:
        - **어텐션 메커니즘**: 다양한 시점의 특징을 종합하는 `Standard10ViewAttention` 모듈 구현.
        - **스케줄러 고도화**: `CosineAnnealingWarmupRestarts`를 도입하여 웜업 단계와 함께 더 정교한 학습률 조절.
        - **증강 파이프라인 최적화**: EDA에서 발견된 Train/Test 데이터셋의 차이를 줄이는 데 초점을 맞춘 `BalancedDocumentTransforms` 구현. 마스킹 증강의 강도를 줄여 정보 손실 최소화.

![2way_modeling](https://github.com/user-attachments/assets/442906e6-17c4-491e-bf62-70e585b00de1)

## 5. Result

### Leader Board
![cv_result](https://github.com/user-attachments/assets/f8dea727-c18a-48c9-8821-4b47d69a2b36)

### Presentation
- https://www.canva.com/design/DAGsyh7kKGw/rbDFnwCmdgDTNvrcs8tJiQ/edit

### Meeting Log
- https://www.notion.so/5-21140cb3731d80949942cdc0f9d2ceae?source=copy_link

### 회고
#### 목표 달성도
이번 프로젝트에서 목표했던 바에 **약 70% 정도 도달**했다고 평가합니다.  
데이터 증강, 다양한 모델 실험 등 가능한 많은 시도를 진행하였고, 제한된 시간 속에서도 의미 있는 결과를 도출할 수 있었습니다.

#### 팀워크 및 잘한 점
- **역할 분배의 명확성**  
매 회의 이후, 각자의 역할을 구체적으로 정리하고 실행한 점이 큰 도움이 되었습니다.

- **의견 충돌 → 발전적 피드백**  
단순히 각자 역할을 수행한 것에서 그치지 않고, 서로 다른 시각에서의 피드백과 논쟁을 통해 더 나은 아이디어를 도출할 수 있었습니다.

- **멘토링 적극 활용**  
위 과정을 통해 사전에 핵심 질문들을 정리했고, 멘토링 시간에 보다 **양질의 피드백**을 받아 다양한 시도를 이어갈 수 있었습니다.

#### 아쉬웠던 점
- **마지막 날 시간 부족**  
실험을 좀 더 정리하고 마무리할 시간이 부족했던 점이 아쉽습니다. 시간만 조금 더 있었다면 성능을 추가로 끌어올릴 수 있었을 것입니다.

- **테스트 전략 부재**  
초기 코드 작성 당시, 학습을 먼저 수행하고 테스트를 나중에 진행하면서 오류가 다수 발생했고, 이로 인해 시간이 많이 소모되었습니다.  
→ `epoch=1`로 우선 테스트를 수행한 뒤 본 학습을 진행했어야 했다는 점을 뼈저리게 느꼈습니다.

- **코드 준비 지연**  
코드 구조 정리가 늦어지며, 다양한 모델 · 옵티마이저 · 스케줄러 · 손실함수 조합 실험까지는 충분히 수행하지 못한 것이 아쉬움으로 남습니다.

#### 향후 계획 및 다짐
향후 유사한 대회에 다시 참여한다면, 이번 경험을 바탕으로  
- **사전 테스트 전략 수립**
- **코드 구조 정비**
- **실험 설계 및 역할 분담 체계화**
를 보다 명확히 하여 **더 높은 완성도와 성능**을 목표로 할 예정입니다.<br>
이번 대회를 통해 **이미지 수집과 라벨링의 중요성**,  <br>그리고 실제 문제 해결에서의 **데이터 기반 접근의 본질**을 실감할 수 있었습니다.

### Reference
- [지난 기수 1등팀 전략 분석](docs/지난%20기수%20cv%201등%20팀%20전략.md)
- [Papers with Code - Document Image Classification](https://paperswithcode.com/task/document-image-classification)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [timm Documentation](https://timm.fast.ai/)
