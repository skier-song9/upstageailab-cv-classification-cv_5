# 📌 Kaggle 수상 전략 분석: DocFormer 기반 문서 이미지 분류
## Document Image Classification with DocFormer

🔗 출처:

[Document Image Classification with DocFormer  💥](https://www.kaggle.com/code/akarshu121/document-image-classification-with-docformer/notebook)

케글 사이트 요약

[Summarize Videos, Audio, PDF & Websites - Lilys AI](https://lilys.ai/digest/4793394/4015207)

---

### 1. **문제 정의**

- **Task**: 문서 이미지 분류 (Document Image Classification)
- **Challenge**: OCR 없이도 이미지에서 시각적 패턴 기반으로 문서 유형을 정확히 분류

### 2. **사용한 모델**

### **DocFormer**

- Microsoft에서 개발한 **문서 이미지 전용 트랜스포머 기반 모델**
- 시각적 + 레이아웃 + 텍스트 정보를 함께 처리 (멀티모달)
- 해당 노트북에서는 HuggingFace의 사전학습된 `microsoft/docformer` 모델 사용

```python
# Notebook Cell #10
from transformers import AutoModel, AutoProcessor
processor = AutoProcessor.from_pretrained("microsoft/docformer")
model = AutoModel.from_pretrained("microsoft/docformer")
```

### 3. **입력 전처리 전략**

- DocFormer는 일반적인 이미지만이 아니라 **OCR bounding box + image + text**를 모두 입력으로 받음
- 하지만 이 노트북에서는 **OCR 없이 시각적 이미지 정보만으로 분류**하는 실험을 진행
- 이미지 리사이징 및 normalization 적용

```python
# Notebook Cell #18
inputs = processor(images=image, return_tensors="pt")
```

### 4. **학습 전략**

- **사전학습 모델만 사용** (docformer-base)
- Linear Classification Head만 학습하도록 설정
- Optimizer: AdamW
- Scheduler: Cosine Annealing

출처: Notebook Cell #25~#30

### 5. **검증 전략**

- K-fold나 Stratified K-Fold 미사용 (단일 validation split)
- 80:20으로 학습/검증 분리

출처: Notebook Cell #13

### 6. 성능 결과

- F1-score 기준 약 **0.86** 성능 달성
- 제출 형식: ID와 예측 클래스(target)를 갖는 CSV

출처: Notebook Output Section

### 7. 실전 응용 가능성 평가

- OCR 없이도 DocFormer가 꽤 높은 성능 달성 → **실제 회전/훼손된 이미지가 많은 환경**에서도 강건할 수 있음
- 입력 텍스트를 포함하지 않기 때문에 **Upstage 대회처럼 변형된 문서 이미지에도 적용 가능성 높음**
- 다만 이 방식은 GPU 메모리 사용량이 높아질 수 있으며, 추론 속도가 느릴 수 있음

## 정리 요약
| 항목 | 내용 |
| --- | --- |
| **사용 모델** | `microsoft/docformer` |
| **텍스트 입력** | 사용하지 않음 (이미지만) |
| **전처리** | 리사이징, Normalization |
| **학습 방식** | Linear Head만 학습 |
| **검증 방식** | 80:20 Split |
| **평가 지표** | F1-Score ≈ 0.86 |
| **적용 가능성** | Upstage 대회에서 회전/훼손 이미지 대응 가능 |