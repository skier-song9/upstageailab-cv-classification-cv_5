# 탐색적 데이터 분석 (EDA) 🔍

-----

## 목차

1.  [밝기 (Brightness) 💡](#밝기-brightness-)
2.  [대비 (Contrast) 🌓](#대비-contrast-)
3.  [흐림 정도 (Blurriness) 🌫️](#흐림-정도-blurriness-️)
4.  [노이즈 추정 (Noise Estimation) 🔊](#노이즈-추정-noise-estimation-)
5.  [가로/세로 종횡비 (Aspect) 📷](#가로세로-종횡비-aspect-)
6.  [회전 추정 (Rotation Estimation) 🔄](#회전-추정-rotation-estimation-)

-----

## 밝기 (Brightness) 💡
- 이미지를 흑백으로 변환하여 픽셀의 중앙값 분포를 파악하여 밝기 분석

| Train | Test |
|---|---|
| ![train bright](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_bright.png) | ![test bright](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_bright.png) |



-----

## 대비 (Contrast) 🌓
- 이미지를 흑백으로 변환하여 픽셀 값의 표준편차를 파악

| Train | Test |
|---|---|
| ![train contrast](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_contrast.png) | ![test contrast](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_contrast.png) |

-----

## 흐림 정도 (Blurriness) 🌫️
- 이미지를 흑백으로 변환하여 Laplacian 필터를 적용한 후, 분산으로 blur 정도를 분석

| Train | Test |
|---|---|
| ![train blur](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_blur.png) | ![test blur](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_blur.png) |

-----

## 노이즈 추정 (Noise Estimation) 🔊
- 웨이블렛 변환을 사용하여 이미지를 다양한 주파수 대역으로 분해, 이미지의 노이즈 표준 편차를 추정

| Train | Test |
|---|---|
| ![train noise](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_noise.png) | ![test noise](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_noise.png) |

-----

## 가로/세로 종횡비 (Aspect) 📷
- 이미지의 너비/높이 비율 분포 분석

| Train | Test |
|---|---|
| ![train aspect](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_aspect.png) | ![test aspect](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_aspect.png) |

-----

## 회전 추정 (Rotation Estimation) 🔄
- Hough Transform을 사용해 이미지에서 직선을 검출하고, 직선들의 기울기를 통해 이미지의 회적 각도를 추정

| Train | Test |
|---|---|
| ![train rotation](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/tree/main/docs/train_rotation.png) | ![test rotation](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/docs/test_rotation.png) |

