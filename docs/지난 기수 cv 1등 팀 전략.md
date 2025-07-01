# 지난 기수 cv 1등 팀 전략

/https://yes-it-spot.tistory.com/entry/Upstage-AI-Lap-5%EA%B8%B0-%EB%B6%80%ED%8A%B8%EC%BA%A0%ED%94%84-%EB%AC%B8%EC%84%9C-%ED%83%80%EC%9E%85-%EB%B6%84%EB%A5%98-CV-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C%EC%97%90%EC%84%9C-1%EC%9C%84-%EB%8F%84%EC%A0%84

### 1. 데이터 분석 및 EDA 전략

| 전략 | 실행 방법 및 인사이트 |
| --- | --- |
| **밝기/대비/노이즈 분석** | - 학습 데이터는 비교적 clean, 테스트 데이터는 **노이즈 + 회전 + 구김** 등 변형 심함- 밝기, 블러, contrast를 정량화하여 분포 확인 |
| **회전 각도 분석** | - Hough Transform 사용- 테스트셋은 다양한 회전값 포함됨- → **RandomRotation 증강 필요** |
| **이미지 크기 분석** | - Train/Test 간 크기 분포차 존재- 전체적으로 test가 더 넓은 분포 가짐 |
| **라벨 분포 분석** | - 클래스 불균형 존재 확인- → 특정 클래스에 CutMix/MixUp 등 증강 집중 |
| **정제 및 품질 검사** | - 오타, 잘못된 라벨링 존재 여부 확인- → 일부 문서 형식 잘못 입력됨 감지 |
| **분류 기준 시각화** | - Confusion Matrix, T-SNE, Class별 예측 경향 분석- → 모델 혼동 구간 파악 및 개선 포인트 탐색 |
| **EDA 주도 역할** | - 단순 분석을 넘어, “이 클래스는 증강 필요” 등 팀 피드백 제공- 프로젝트 전체 방향 설정에 핵심 기여 |

---

### 2. 데이터 증강 전략

| 항목 | 내용 |
| --- | --- |
| **Off-line 증강** | 학습 전 증강된 이미지 데이터셋 생성 (rotation, crop 등 반영) |
| **On-line 증강** | 학습 시 실시간 augmentation 적용- `RandomRotation`, `BrightnessJitter`, `ContrastAdjustment`, `MixUp`, `CutMix` |
| **클래스별 증강 조절** | - 소수 클래스에 대해 증강 집중 적용 |
| **TTA (Test Time Augmentation)** | - 앙상블 성능 강화 및 예측 안정성 확보 |

---

### 3. 모델 실험 전략

| 항목 | 시도한 모델 |
| --- | --- |
| **CNN 계열** | - ResNet50 / EfficientNet / EfficientNetV2- CustomCNN / ConvNeXt / CoAtNet |
| **Transformer 계열** | - Swin Transformer V2: 문서 전체 구조 반영에 유리- MaxViT: CNN + Transformer 융합 구조- LayoutLMv3: 실제 문서 레이아웃 모델 실험 |
| **Contrastive Learning** | - SimCLR (ResNet50 backbone 기반) |
| **성능 기준** | - ConvNeXt: F1 ≈ 0.995 / 리더보드 ≈ 0.96- Swin V2: F1 ≈ 0.999 / 리더보드 ≈ 0.937- MaxViT: F1 ≈ 0.997 / 리더보드 ≈ 0.959- SimCLR: F1 ≈ 0.967 / 리더보드 ≈ 0.847 (낮음) |

---

### 4. 모델 튜닝 및 하이퍼파라미터 조정

| 항목 | 적용 기법 |
| --- | --- |
| **Optimizer** | Adam / AdamW |
| **Learning Rate Scheduler** | CosineAnnealingLR, OneCycleLR |
| **Early Stopping** | 적용하여 과적합 방지 |
| **Loss Function** | CrossEntropyLoss 추정 |
| **Batch Size / Epoch** | 모델 구조 및 증강에 따라 조정 |
| **W&B 사용** | 실험 로깅 및 시각화 공유 자동화 |
| **실험 조건 정리** | 실험 순번, 증강 조건, Best 모델 체크, TTA 적용 여부 등 모두 명시적으로 기록하여 팀 내 공유 |

---

### 5. 앙상블 전략

| 전략 | 설명 |
| --- | --- |
| **다양한 모델 앙상블** | - CNN 계열 + Transformer 계열 결과 Soft Voting |
| **TTA + 앙상블** | - TTA 결과들 포함해 예측 안정성 높임 |
| **클래스별 예측 혼동 분석 기반** | - 혼동 클래스에 대해 앙상블이 더 좋은 선택을 하도록 설계 |

---

### ✅ 팀 전략의 특징 요약

| 포인트 | 설명 |
| --- | --- |
| **1. EDA 기반 설계** | 단순 시각화가 아닌, 증강/모델 선택까지 연계하는 분석 |
| **2. 증강 전략 다양화** | 오프라인 + 온라인 병행, MixUp/CutMix, 회전 대응 특화 |
| **3. 다양한 Backbone 탐색** | CNN부터 Transformer까지 문서 특성에 맞게 실험 |
| **4. 하이퍼파라미터 관리** | W&B 등 도구 기반 반복 실험 최소화 |
| **5. 앙상블 최적화** | 강한 모델 조합 및 예측 안정성 확보 목적 |

---

### 📊 정량적 성과 (발췌)

| 모델 | Val F1 | 리더보드 점수 |
| --- | --- | --- |
| ConvNeXt | 0.995 | 0.9649 |
| CoAtNet | 0.9975 | 0.9351 |
| Swin V2 | 0.9991 | 0.9373 |
| MaxViT | 0.9976 | 0.9590 |
| SimCLR | 0.9672 | 0.8474 |

---

이 팀의 전략은 실전 CV 분류 문제에서 모범적인 방식입니다. 특히:

- EDA → 증강 → 모델링 → 튜닝 → 앙상블로 이어지는 **End-to-End 전략 전개**
- **문서 특성에 맞춘 증강 및 Transformer 구조 실험**
- 실험 로깅, 협업, 피드백 반영 등 **팀워크 기반의 반복 실험 구조**

가 매우 인상적인 포인트였습니다.