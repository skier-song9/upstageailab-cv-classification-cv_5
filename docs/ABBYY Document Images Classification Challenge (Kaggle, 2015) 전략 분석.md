# 📌 ABBYY Document Images Classification Challenge (Kaggle, 2015) 전략 분석

> ⚠️ **주의사항**: 본 전략 요약은 ABBYY Document Images Classification Challenge (2015)에 대해 **공식적으로 공개된 우승 솔루션이 없으며**, 
대회 역시 **초청 기반(private)**으로 운영되었기 때문에, 당시의 기술적 배경과 일반적인 문서 이미지 분류 접근법에 근거하여 **논리적으로 재구성된 추정 전략**입니다. 
실제 사용된 방식과 일부 차이가 있을 수 있습니다.

---

## 대회 개요

| 항목 | 설명 |
|------|------|
| **주최** | ABBYY (OCR 전문 기업) |
| **대회 유형** | 초청 기반 이미지 분류 경진대회 |
| **목표** | 스캔 문서 이미지(계약서, 송장, ID 등)를 클래스별로 정확히 분류 |
| **클래스 수** | 16개 문서 타입 |
| **데이터 규모** | 약 90,000장 (Train: 44,000 / Test: 45,000) |
| **파일 형식** | JPG, PNG 등 (OCR 텍스트는 미제공) |

---

## 📁 데이터 특징

| 항목 | 내용 |
|------|------|
| **포맷** | 이미지 (PNG/JPG), OCR 텍스트 없음 |
| **문서 유형** | 계약서, 송장, 세금 문서, 신분증 등 |
| **해상도** | 대체로 200dpi 이상, 일부 흐림/회전/노이즈 있음 |
| **레이아웃** | 클래스 간 시각적 구조 및 배치 차이 뚜렷 |
| **노이즈 요소** | 접힘, 그림자, 배경 노이즈, 기울기 등 포함 가능성 |

---

#✅ 핵심 전략 요소 정리 (기술 추정 기반)
전략 요소	설명
데이터 전처리	
- Binarization (adaptive thresholding)
- Aspect Ratio 유지 + Padding
- Contrast Enhancement
- Deskew (기울기 보정)
이미지 증강	
- Rotation (±10~15도)
- Random Crop
- Gaussian Blur
- Brightness / Contrast 조정
- Noise Injection (Salt-and-pepper 등)
모델링 전략	
- CNN 기반 (VGGNet, GoogLeNet, AlexNet 등)
- Transfer Learning: Feature Extractor 활용
- Fine-tuning (부분 레이어만)
- 일부 SVM 분류기 후단에 연결 (CNN+SVM 구조)
텍스트 기반 전략 (OCR)	
- OCR 텍스트 추출 (ABBYY OCR or Tesseract)
- Keyword 추출 + TF-IDF
- Text Embedding (Word2Vec 등)
- 텍스트 기반 분류기 (SVM, Naive Bayes 등)
멀티모달 융합	
- Early Fusion: 이미지 특징 + 텍스트 특징 결합 후 분류
- Late Fusion: 텍스트 기반 모델과 이미지 기반 모델 결과 앙상블
- 병렬 네트워크 구조 (CNN + MLP 등)
학습 최적화	
- Optimizer: SGD with Momentum, Adam
- Learning Rate Decay (step decay)
- Regularization: Dropout, L2 Weight Decay
- K-Fold Cross Validation
앙상블 기법	
- 서로 다른 CNN 구조 앙상블
- 동일 구조에서 초기값/하이퍼파라미터 다르게 학습한 모델 앙상블
- 이미지 기반 + 텍스트 기반 모델 결합 (Late Fusion)
평가 지표	
- Primary: Accuracy
- Secondary: Precision/Recall by class (시각화)

# 우리가 현재 적용할 수 있는 전략
적용 전략	설명
다양한 이미지 증강: 	
- 회전, 노이즈 등으로 현실적 변형에 강건한 모델 학습
문서 레이아웃 기반 증강:	
- 문서 구조 인식력 강화 (corner masking, flip 등)
CNN 기반 모델 스태킹:	
- ConvNeXt, Swin 등 최신 CNN으로 교체 가능
비율 유지 + 패딩 Resize:	
- 문서 왜곡 없이 모델 입력 형식 통일

---

## 정리: 이 대회의 전략적 의의

- OCR 기업인 ABBYY가 주최했기 때문에 **OCR + 이미지 기반 멀티모달 접근**이 주요 전략이었을 가능성 높음
- 당시 최신 기술이었던 CNN (VGGNet, GoogLeNet)을 적극 활용했을 것으로 추정
- 다양한 전처리, 증강, 앙상블 기법은 **지금도 강력한 baseline 전략**으로 재해석 가능
- OCR을 활용하지 않더라도, **레이아웃 중심의 이미지 분류 전략**만으로도 성능 확보 가능

---

## 🔗 참고 링크

- [Kaggle 대회 링크 (ABBYY Challenge)](https://www.kaggle.com/competitions/abbyy-documents-images-classification-challenge)
- [ImageNet 2014 우승 모델 참고](https://paperswithcode.com/sota/image-classification-on-imagenet)