# ◾OOM
- Out-of-Memory 를 지칭하는 용어로, 모델의 parameter 개수가 많거나 이미지의 해상도가 높으면서 batch_size가 클 때, 하나의 batch를 GPU가 감당하지 못하는 경우에 발생하는 오류이다. 
- 원인 : 너무 큰 모델 크기, 너무 큰 image 해상도, 너무 큰 batch_size

# ◾해결방법

## 🔻Mixed Precision 학습
### 🔸개념
- 모델 파라미터의 데이터 타입을 변경하는 것이 아니라, 순전파~역전파 과정에서 tensor의 데이터 타입을 FP16 또는 BF16으로 자동 변환하여 연산함으로써 메모리 사용 효율을 높이고 성능 손실을 최소화하는 방법이다.
    - 모델 파라미터의 데이터 타입은 여전히 `FP32`이다.
    - `with autocast():` block 내의 코드에서 연산되는 tensor들은 데이터 타입이 FP16, BF16으로 자동 변환된다.
    - FP16/BF16의 표현 범위 문제로 인해 발생할 수 있는 수치적 불안정성(예: 그라디언트 underflow)을 `GradScaler`가 FP32 파라미터와 연동하여 관리하므로 FP32에 가까운 성능을 유지할 수 있다.
- 지수(Exponent) 비트: 숫자의 크기(동적 범위)를 결정합니다. 지수 비트가 많을수록 더 넓은 범위의 수를 표현할 수 있습니다.

    - FP16은 지수 비트가 5개이므로 표현할 수 있는 범위가 6.1 × 10^-5에서 6.5 × 10^4 정도로 매우 좁습니다.

    - BF16은 지수 비트가 8개로, 이는 FP32와 동일합니다. 따라서 1.2 × 10^-38에서 3.4 × 10^38까지 FP32와 유사한 넓은 동적 범위를 가집니다.

- 가수(Mantissa) 비트: 숫자의 정밀도를 결정합니다. 가수 비트가 많을수록 더 정확하게 숫자를 표현할 수 있습니다 (소수점 이하 자릿수).

    - FP16은 가수 비트가 10개로, BF16보다 정밀도가 높습니다.

    - BF16은 가수 비트가 7개로, FP16보다 정밀도가 낮지만 FP32와 동적 범위가 같기 때문에 딥러닝 학습 시 발생하는 극단적인 값(아주 작거나 아주 큰 그라디언트) 문제에 더 강건합니다.

- **FP16의 문제점**: 좁은 동적 범위 때문에, 역전파 과정에서 계산된 그라디언트가 너무 작아져 FP16으로 표현할 수 있는 최소값(6.1 × 10^-5)보다 작아지면, 그 값이 0으로 잘려버리는 Underflow (언더플로우) 문제가 발생할 수 있습니다. 모든 그라디언트가 0이 되면 모델이 더 이상 학습되지 않습니다. 반대로 값이 너무 커지면 **Overflow (오버플로우)**가 발생하여 inf (무한대)가 될 수도 있습니다.
    - **BF16의 장점**: FP32와 동일한 동적 범위를 가지므로, Underflow나 Overflow 문제가 훨씬 적습니다. 이는 학습 안정성을 크게 높여주고, FP32에서 BF16으로 전환할 때 코드를 거의 수정할 필요가 없게 만듭니다.

### 🔸코드
- ⚠️ deprecated : `from torch.cuda.amp import autocast, GradScaler` 가 최신 torch에서는 deprecated될 수 있다. 이번 프로젝트의 버전인 2.5.1에서는 괜찮다.
- Training 코드 : optimizer.zero_grad() 이후에 `autocast` 추가

```python
for train_x, train_y in self.train_loader: # batch training
    train_x, train_y = train_x.to(self.cfg.device), train_y.to(self.cfg.device)
    
    self.optimizer.zero_grad() # 이전 gradient 초기화

    if self.cfg.mixed_precision: # FP16을 사용해 메모리 사용량 감소
        # autocast 컨텍스트 매니저 사용
        with torch.cuda.amp.autocast():
            outputs = self.model(train_x)
            loss = self.criterion(outputs, train_y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update() # 다음 반복을 위해 스케일 팩터를 업데이트

    else: # 일반적인 학습 시
        outputs = self.model(train_x)
        loss = self.criterion(outputs, train_y)
        loss.backward() # backward pass
        self.optimizer.step() # 가중치 업데이트
```

- Validation 코드 

```python
with torch.no_grad():  # gradient 계산 비활성화
    for val_x, val_y in self.valid_loader: # batch training
        val_x, val_y = val_x.to(self.cfg.device), val_y.to(self.cfg.device)
        
        if self.cfg.mixed_precision: # FP16을 사용해 메모리 사용량 감소
            # autocast 컨텍스트 매니저 사용
            with torch.cuda.amp.autocast():
                outputs = self.model(val_x)
                loss = self.criterion(outputs, val_y)
        else:
            outputs = self.model(val_x)
            loss = self.criterion(outputs, val_y)
```

## 🔻GPU cache 비우기
- training step, validation step 마다 GPU에 할당된 batch 데이터를 제거하고, gpu cache를 비워주는 코드를 실행한다.
- `for batch_x, batch_y in self.valid_loader` for 반복문 안에서 아래 코드를 호출해야 한다.

```python
del val_x, val_y, outputs, loss # 사용된 변수 명시적 삭제
torch.cuda.empty_cache() # cuda GPU 캐시 비우기
```