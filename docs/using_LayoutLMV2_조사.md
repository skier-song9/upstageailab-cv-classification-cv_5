# Document Classification:: LayoutLMV2

https://www.kaggle.com/code/anantgupt/document-classification-layoutlmv2#Infrence

## 논문 및 경진대회의 데이터 이미지

데이터셋 설명 및 다운 링크 : https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg/data

- 10개 클래스

- jpg 문서 이미지

- 각 클래스별 이미지 수가 121~621개로 이루어짐(데이터 불균형)

- 노이즈가 많은 문서, 각도가 조금 틀어진 문서, 가로로 누워있는 문서도 있음

## 논문 및 경진대회의 전략

### 이미지 전처리 및 분석
- EDA를 어떻게 했는가?
    - 각 클래스별 이미지 개수를 글자와 bar 그래프 형태로 출력
    
    - 각 클래스별로 랜덤한 이미지 하나씩 시각화

- 어떤 전처리 기법을 사용?
    - 클래스명과 숫자 인코딩 값 간의 변환을 쉽게 하기 위한 매핑 딕셔너리 총 2개 생성

    - 이미지명과 라벨명을 한 번에 모아 DataFrame으로 정리

    - 열리지 않는 이미지를 DataFrame에서 삭제

    - 데이터셋 분할 : train_test_split()을 사용하여, 8:2 비율로 학습, 테스트 데이터셋 분할(분할 후, 각 데이터셋의 클래스별 데이터 개수 출력)

    - 문서 이미지를 LayoutLMv2 입력 형태로 변환하는 Processor를 생성하여 사용
    
        변환 전: 경로와 라벨 문자열 (**image_path**, **label**)

        변환 후:

        **image** : (3, H, W) Tensor (리사이즈된 문서 이미지)

        **input_ids** : [512] Tensor (토큰 ID)

        **attention_mask** : [512] Tensor (마스크)

        **token_type_ids** : [512] Tensor (세그먼트 구분)

        **bbox** : (512, 4) Tensor (토큰별 bounding box)

        **labels** : 정수 라벨 Tensor

### 사용한 모델
문서 이미지 + 텍스트 + 레이아웃(위치) 정보를 동시에 활용해 문서 이해(Document Understanding)를 수행하는 멀티모달 Transformer 모델인 LayoutLMv2

### 데이터 증강
- 증강 기법, augmentation 미사용

### Train, Test 전략
- 전이학습을 활용하였고, 보편적인 학습 코드임
- 사전학습된 분류모델을 가져와 AdamW 옵티마이저를 적용하여 train과 test 진행
- lr = 5e-5, epoch = 20, batch_size = 8

### 결과 및 특징
- epoch 7을 기점으로 최적 지점을 벗어난 듯 보임

- epoch 6에서 최고 성능
    
    - train : loss = 0.1592, acc = 96.01

    - test : loss = 0.2544, Accuracy = 92.25

- epoch 0 ~ 6 : 
    - train 성능 ⬆️
    - test 성능 ⬆️
- epoch 7 :
    - Train 성능 계속 ⬆️
    - Test 성능 하락 시작
- epoch 8 ~ 19 :
    - Train 성능 ⬇️
    - Test 성능 ⬇️

### 해결 방법
- Early Stopping 도입

- Learning Rate Scheduler 사용

- 학습률(LR) 낮추어 재학습

- Regularization 강화

    - Dropout 비율 소폭 증가

    - Weight Decay 확인 (AdamW 사용 시 weight_decay 적절 설정)

- Batch Size 작게 조정
