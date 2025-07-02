# 탐색적 데이터 분석 (EDA) 🔍

-----

## 목차

1.  [밝기 (Brightness) 💡](#밝기-brightness-)
2.  [대비 (Contrast) 🌓](#대비-contrast-)
3.  [RGB 채널별 색상 분포 (Color Channel) 🌈](#rgb-채널별-색상-분포-color-channel-)
4.  [흐림 정도 (Blurriness) 🌫️](#흐림-정도-blurriness-)
5.  [노이즈 추정 (Noise Estimation) 🔊](#노이즈-추정-noise-estimation-)
6.  [가로/세로 종횡비 (Aspect) 📷](#가로세로-종횡비-aspect-)
7.  [회전 추정 (Rotation Estimation) 🔄](#회전-추정-rotation-estimation-)
8.  [기타](#기타)

-----

## 밝기 (Brightness) 💡
- 이미지를 흑백으로 변환하여 픽셀의 중앙값 분포를 파악하여 밝기 분석

| Train | Test |
|---|---|
| ![train bright](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_bright.png) | ![test bright](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_bright.png) |

> 해석 : test 데이터셋이 train 데이터셋보다 평균적으로 20정도 더 밝다. 아래 Color Channel에서 코드를 같이 다룬다.

-----

## 대비 (Contrast) 🌓
- 이미지를 흑백으로 변환하여 픽셀 값의 표준편차를 파악

| Train | Test |
|---|---|
| ![train contrast](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_contrast.png) | ![test contrast](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_contrast.png) |

> 해석 : 평균, 분산, 표준편차에 큰 차이가 없다.
> ColotJitter에서 큰 변화를 주지 않아야 한다.

-----

## RGB 채널별 색상 분포 (Color Channel) 🌈
- 이미지의 RGB 채널별 픽셀값의 분포를 분석

| Train | Test |
|---|---|
| ![train color channel](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_colorchannel.png) | ![test color channel](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_colorchannel.png) |

> 해석 : test 데이터셋의 RGB 분포가 평균적으로 20 정도 오른쪽으로 치우쳤다. ColorJittering에 반영해야 한다.
```python
A.ColorJitter(brightness=0.1, contrast=0.07, saturation=0.07, hue=0.07, p=1.0)
```
- `brightness=0.1` : 보정 계수를 `[max(0, 1 - brightness), 1 + brightness]`로 무작위 샘플링한 후, RGB 채널에 각각 보정 계수를 곱하여 픽셀 값을 변화시킨다. > 픽셀값을 대략 +20 ~ -20 하는 효과를 준다.
- `contrast`, `saturation` : 대비, 채도 변화는 최소화한다.
- `hue=0.07` : RGB 분포가 전체적으로 20 씩 차이나고 있으므로 색조 변화가 크기보단 밝기 변화에 해당함. 색조 변화도 최소화하여 적용한다.

-----

## 흐림 정도 (Blurriness) 🌫️
- 이미지를 흑백으로 변환하여 Laplacian 필터를 적용한 후, 분산으로 blur 정도를 분석

| Train | Test |
|---|---|
| ![train blur](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_blur.png) | ![test blur](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_blur.png) |

> 해석 : Laplacian variance가 작을수록 선명도가 낮아진다. test의 Laplacian variance가 더 좌측에 치우쳤으며 더 첨도가 작다. 이는 **test 데이터셋에 선명한 이미지가 train보다 현저히 적으며**, 특정 수준의 흐릿한 이미지가 지배적이다(아마 Mixup의 영향이 있지 않을까 싶다).
```python
# blur_limit 범위는 테스트 분포의 주요 피크 영역을 참고하여 조절합니다.
# 0~1000 사이의 낮은 라플라시안 분산 값(강한 블러)이 많으므로,
# 비교적 강한 블러도 포함되도록 sigma_limit나 blur_limit을 설정하는 것이 좋습니다.
A.OneOf([
    A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0), # sigma를 2.0까지 늘려 더 강한 블러 포함
    A.Blur(blur_limit=(3, 8), p=1.0), # 커널 크기를 8x8까지 늘려 더 강한 블러 포함
    # MedianBlur는 여기서는 제외하거나, 블러 효과만을 원한다면 사용
], p=0.3), # 30% 확률로 둘 중 하나의 블러 적용
```

-----

## 노이즈 추정 (Noise Estimation) 🔊
- 웨이블렛 변환을 사용하여 이미지를 다양한 주파수 대역으로 분해, 이미지의 노이즈 표준 편차를 추정

| Train | Test |
|---|---|
| ![train noise](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_noise.png) | ![test noise](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_noise.png) |

> 해석 : Wavelet 변환을 통해 구한 통계값으로 추정한 노이즈 분포는 Train에서 더 첨도가 작고 분포도 왼쪽으로 더 치우쳐있다. 이는 test 데이터셋의 노이즈 강도가 더 강함을 의미한다.
```python
# Test 데이터의 노이즈 범위가 넓고 강한 노이즈도 포함하므로,
# var_limit의 상한을 어느 정도 높게 설정하여 다양한 강도의 노이즈를 추가합니다.
# p는 모든 이미지에 노이즈를 추가할 필요는 없으므로 0.5~0.7 정도로 설정할 수 있습니다.
A.GaussNoise(std_range=(0.0025, 0.2), p=1.0),, # p 확률로 가우시안 노이즈 적용
                                            # var_limit (분산) 범위는 실험적으로 조정 필요
                                            # (10.0, 70.0)은 일반적인 이미지에서 눈에 띄는 노이즈를 생성할 수 있는 범위
```
- Train 데이터셋보다 Test 데이터셋에서 노이즈가 강한 이미지가 약 60% 많으므로 70% 확률로 wavelet 변환 노이즈 통계값이 0.01~0.03이 나오도록 std_range를 0.01, 0.035로 하여 노이즈를 적용한다.
- 그러나 육안으로 Test 데이터셋을 보았을 때 noise_std가 0.2~0.3에 해당할 것 같이 노이즈가 심한 데이터도 있었다. 따라서 std_range는 (0.0025, 0.2)로 최종 결정하고 p=1로 한다.

-----

## 가로/세로 종횡비 (Aspect) 📷
- 이미지의 너비/높이 비율 분포 분석

| Train | Test |
|---|---|
| ![train aspect](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_aspect.png) | ![test aspect](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_aspect.png) |

> 해석 : Train에서는 0.7, 1.3~1.5 종횡비가 가장 많았고, Test에서도 마찬가지였는데 4:3 종횡비가 급증했다.    
> Multi-Crop(multi-view) training 또는 inference가 도움이 될 수 있다.   
> ~~또는 RandomCrop을 적용~~
```python
A.LongestMaxSize(max_size=CFG.image_size),
A.PadIfNeeded(min_height=CFG.image_size, min_width=CFG.image_size,
    border_mode=cv2.BORDER_CONSTANT, fill=(255,255,255), p=1.0
)
```
- 최신 모델들은 multi-scale feature map을 추출하는 능력이 뛰어나므로 별도의 spp layer는 필요 없을 듯 하다. train 데이터셋의 해상도가 보통 이상이라는 점을 고려해 CFG.image_size를 384 이상으로 가져가고, 정사각형으로 만들 때 여백은 흰색으로 채운다.

-----

## 회전 추정 (Rotation Estimation) 🔄
- Hough Transform을 사용해 이미지에서 직선을 검출하고, 직선들의 기울기를 통해 이미지의 회적 각도를 추정

| Train | Test |
|---|---|
| ![train rotation](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/train_rotation.png) | ![test rotation](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/blob/main/docs/images/test_rotation.png) |

> 해석 : 히스토그램과 달리 0~360도의 다양한 회전이 적용됨. 그러나 Transpose 때문에 rotate을 심하게 주면 안 됨.
```python
A.ShiftScaleRotate(shift_limit=(-0.05,0.05),
    scale_limit=(-0.15, 0.15), 
    rotate_limit=(-20, 30),
    fill=(255,255,255), 
    p=0.9
),
A.Transpose(p=0.5)
```
- Shift도 약간 추가하기 위해 `ShiftScaleRotate`를 사용. Shift와 Scale은 약간의 변화만 주고 Rotate에 중점을 둔다.
- Transpose 가 적용된 것으로 예상된다. 

-----

## 기타

- 그림자 : 인조적인 그림자는 없다. 사진 찍는 과정에서 발생하는 자연스러운 그림자만 존재
- Mixup : 다수
- CutMix : 소수
- 질감 변화 : 소수의 이미지에서 이미지의 일부분의 질감을 (가죽 등)의 형태로 변화시킨다.
...