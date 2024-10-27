---
title: "시각지능 딥러닝 | Visual Intelligence Deep Learning" 
date: 2024-10-27 12:13:24 +0900
achieved: 2024-10-25 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Visual Intelligence]
---
----------
> KT 에이블스쿨 6기 시각지능 딥러닝에 진행한 강의 내용 정리한 글입니다. 
{: .prompt-info } 

## **Computer Vision**
- OCR (Optical Character Recognition): 문서의 텍스트를 인식하고 식별
- Vision Biometrics: 홍채 패턴 인식을 통해 사람들을 구분
- Object Recognition: 실시간의 이미지나 스캔 입력을 가지고 제품을 분류
- Special Effects: 모션 캡처 및 모양 캡처, 영화에서의 CGI
- 3-D Printing and Image Capture: 영화, 건축 구조 등에 사용
- Sports: 경기에서 필드에 추가 라인을 그리거나 이를 기반으로 영상 판독
- Social Media: 얼굴 모양에 맞추어 착용되는 이미지
- Smart Cars: 사물과 사람을 인지
- Medical Imaging: 3D 이미징 및 이미지 유도 수술

## **CNN**

![CNN](https://github.com/user-attachments/assets/914e7cd4-37d1-45da-99c7-750cc8166847)

CNN(Convolutional Neural Network)은 이미지 처리와 패터 인식에 주로 사용되는 신경망 구조로 Convolutional layer, Pooling layer, Fully-connected layer 구성됩니다. 

- Convolutional layer
    - 이미지 특징을 추출하는 역할
    - 필터 또는 커널이라고 불리는 작은 크기의 행렬을 사용해 이미지에 대해 합성곱 연산을 수행
    - 이 과정을 통해 이미지를 각 위치의 특징 맵(Feature Map)으로 변환하여 학습에 중요한 정보를 추출
        - 패딩(Padding): 합성곱 연산 후 출력 이미지의 크기를 원본 이미지 크기로 유지하거나 출력 크기를 조정하는 방법
        - 스트라이드(Stride): 필터를 적용할 때의 이동 간격을 뜻하며 스트라이드가 클수록 출력 크기가 작아지고 연산 속도가 빨라집니다. 
- Pooling layer
    - 이미지의 크기를 줄여 계산량을 감소시키고 과적합 방지 역할
        - 최대 풀링(Max Pooling): 일정 구역 내에서 최대값을 추출하여 중요한 특징을 유지하며 차원을 축소
        - 평균 풀링(Average Pooling): 구역 내에서 평균값을 추출하여 특징을 보존하면서 차원을 축소
- Fully-connected layer 
    - 합성곱 레이어와 풀링 레이어를 거쳐 얻어진 다차원 특징 맵은 일렬로 나열되어 1차원 형태의 벡터로 변환되는데 이를 Flattening이라고 합니다. 
    - 최종 분류 작업을 수행하며 이전 레이어에서 추출한 특징들을 종합하여 예측
    
### **LeNet-5**

![LeNet-5](https://github.com/user-attachments/assets/b8e0b49b-35d2-40c4-81d4-ff197b2b50f8)

LeNet-5는 손글씨 숫자 인식(MNIST 데이터셋)과 같은 간단한 이미지 분류 작업에 주로 사용됩니다. 
- Convolutional layer(C) 3개와 Sub-Sampling Layer(S) 2개, Fully-Connected layer(F) 2개로 구성
- C1 → S2 → C3 → S4 → C5 → F6 → F7(output) 구조
    - C1: 6개의 5x5 필터를 사용하여 28x28 크기의 특징 맵 생성
    - S2: 평균 풀링(Average Pooling)을 사용하여 14x14 크기로 차원 축소 
    - C3: 16개의 5x5 필터를 사용하여 10x10 크기의 특징 맵 생성
    - S4: 다시 평균 풀링을 적용하여 5x5 크기로 차원 축소
    - C5: 120개의 5x5 필터를 사용하여 1x1 크기의 특징 맵 생성
    - F6: 84개의 노드로 구성, 이전 레이어의 출력이 이 레이어에 연결
    - F7: 10개의 노드로 구성되어 0부터 9까지의 숫자에 대한 확률을 출력 

### **AlexNet**

![AlexNet](https://github.com/user-attachments/assets/0a413cb1-53a8-41bd-8e99-3ddf3760877e)

- Convolutional layer(C) 5개와 Sub-Sampling Layer(S) 3개, Fully-Connected layer(F) 3개로 구성
- C1 → S2 → C3 → S4 → C5 → C6 → C7 → S8 → F9 → F10(output) 구조
    - C1: 96개의 11x11 필터를 사용하여 55x55 크기의 특징 맵을 생성
    - S2: 최대 풀링(Max Pooling)을 적용하여 27x27 크기로 차원 축소
    - C3: 256개의 5x5 필터를 사용하여 27x27 크기의 특징 맵을 생성
    - S4: 다시 최대 풀링을 적용하여 13x13 크기로 차원 축소
    - C5: 384개의 3x3 필터를 사용하여 13x13 크기의 특징 맵을 생성
    - C6: 384개의 3x3 필터를 사용하여 13x13 크기의 특징 맵을 생성
    - C7: 256개의 3x3 필터를 사용하여 13x13 크기의 특징 맵을 생성
    - S8: 최대 풀링을 적용하여 6x6 크기로 차원 축소
    - F9: 4096개의 노드로 구성, 이전 레이어의 출력을 종합하여 처리
    - F10: 4096개의 노드로 구성, Dropout 기법을 사용하여 과적합 방지 
        - 최종 출력층은 F10과 연결되며 1000개의 노드로 구성
        - 각 노드는 1000개의 클래스에 대한 확률을 출력

### **Feature map**
$$
n_{out} = \left\lfloor \frac{n_{in} + 2p - k}{s} \right\rfloor + 1
$$
- $n_{in}$: input feature map의 가로세로 사이즈
- $n_{out}$: output feature map의 가로세로 사이즈
- $k$: Convolution filter의 가로세로 사이즈
- $s$: Convolution filter의 이동포복
- $p$: Convolution filter map에 덧붙일 pad의 수

## **Data Augmentation**
모델의 일반화 능력을 향상시키고, 과적합을 방지하기 위해 훈련 데이터의 양을 인위적으로 늘리는 기법
- 데이터를 추가로 수집하기에 어려운 경우 현재 가지고 있는 데이터에 Augmentation을 진행하여 데이터를 증식
- 크기, 반전, 이동, 회전, 자르기 등의 방법이 존재

## **Transfer Learning**
사전 훈련된 모델의 가중치를 새로운 작업에 적용하는 방법
- 데이터셋이 작거나 모델 훈련에 필요한 시간이 긴 경우에 유용하게 사용
- 새로운 작업에 특화된 레이어를 추가하거나 기존 레이어를 미세 조정하여 전이 학습을 수행

## **Object Detection**
객체 탐지는 이미지나 비디오에서 특정 객체를 인식하고 해당 객체의 위치를 나타내는 방법
- Localization: 이미지에서 단일 객체의 위치를 찾는 과정 
- Object Detection: 이미지 내에서 여러 개의 객체를 동시에 인식하고 그 위치를 찾는 과정 

### **Bounding Box**
이미지 내에서 객체의 위치를 네 개의 좌표를 이용해서 사각형으로 감싸는 방법
- 이미지 내 객체의 좌표를 표시하고 크기와 위치 정보를 제공

### **Class Classification**
객체가 어떤 종류에 속하는지 클래스를 예측하는 작업
- 객체 검출 모델은 이미지 안의 여러 객체를 탐지하고 각 객체의 클래스를 분류

### **Confidence Score**
해당 클래스에 속한다고 확신하는 정도를 수치로 표현
- 0 ~ 1 사이의 값을 가지며 1에 가까울수록 예측이 정확하다고 확신하는 수준이 높다는 뜻
- 특정 클래스와 Bounding Box의 위치 정확도에 신뢰도 
    - 단순히 Object가 있을 확률
        - 객체가 해당 위치에 존재할 가능성만 평가
    - Object가 있을 확률 X IoU
        - 객체가 존재할 가능성에 IoU를 곱하는 방식
        - IoU는 예측한 Bounding Box와 실제 Bounding Box가 겹치는 정도를 측정하는 지표로 값이 1에 가까울수록 정확하게 예측된 위치를 의미
    - Object가 특정 클래스일 확률 X IoU
        - 객체가 특정 클래스에 속할 확률과 IoU를 곱하는 방식

#### **IoU**
$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

- Area of Overlap: Ground-truth Bounding Box와 Prediction Bounding Box가 겹치는 영역의 면적
- Area of Union: 두 박스의 합집합 영역의 면적
- IoU 계산 과정
    1. 겹치는 영역을 먼저 구해 겹치는 부분의 면적(Overlap)을 계산
    2. 두 박스의 합집합 영역을 구함
        - 합집합 영역은 Ground-truth Bounding Box의 면적 + Prediction Bounding Box의 면적 - Overlap 영역으로 계산
    3. Overlap 영역을 Union 영역으로 나눠 IoU 값을 구함
- 0 ~ 1 사이의 값을 가지며 값이 1에 가까울수록 모델의 예측이 정확

### **NMS**
객체 감지에서 중복된 바운딩 박스를 제거하여 최종적으로 가장 신뢰도 높은 바운딩 박스를 선택하는 기법
1. Confidence Score 임계값 적용 
    - 모델이 예측한 바운딩 박스 중 신뢰도가 낮은 것(Confidence Score가 임계값보다 낮은 것)은 제거
2. Confidence Score 내림차순 정렬
    - 남은 바운딩 박스를 Confidence Score 순서대로 내림차순 정렬
    - Confidence Score가 높은 박스를 우선으로 선택할 수 있게 준비하는 과정
3. 박스 제거 조건 설정
    - Confidence Score가 가장 높은 바운딩 박스를 기준으로, 이 박스와 겹치는 다른 박스들과의 IoU 값을 계산
    - IoU 임계값을 기준으로, IoU가 높아 중복된 영역을 가지는 다른 박스들은 제거
4. 과정 반복
    - 1 ~ 3번 단계를 반복하여 남은 박스 중 Confidence Score가 높은 것부터 확인하며 IoU 임계값 이상인 박스를 제거
    - 이 과정을 통해 결국 Confidence Score가 높은 단일 박스만 남음
5. 최종 결과
    - 한 객체에 대해 하나의 바운딩 박스만 남게 되므로 객체가 한 번만 검출되는 결과 

#### **임계값**
NMS에서 불필요한 바운딩 박스를 필터링하기 위해 설정하는 기준값을 의미
- Confidence Score 임계값
    - 모델이 예측한 바운딩 박스의 신뢰도를 평가하는 기준
    - Confidence Score가 임계값보다 높은 박스만을 유지하고, 낮은 박스는 제거하여 노이즈와 불필요한 박스를 필터링
- IoU 임계값
    - 중복되는 바운딩 박스들 중에서 겹침 정도를 기준으로 불필요한 박스를 제거하는 기준
    - Confidence Score가 높은 상위 박스와 다른 박스들 간의 IoU를 계산하여 IoU가 임계값보다 높은 박스는 동일 객체로 간주하고 제거 

## **Confusion Matrix** 
![Confusion Matrix](https://github.com/user-attachments/assets/80d8c072-2d81-4687-bae7-a58776d99a47)

- Confusion Matrix with O.D
    - TP: 실제 Object를 모델이 Object라 예측
        - 모델이 올바르게 탐지 
    - FP: Object 아닌데 모델이 Object라 예측
        - 모델의 잘못된 탐지
    - FN: 실제 Object를 모델이 아니라고 예측
        - 모델의 잘못된 탐지
    - TN: Object 아닌데 모델도 아니라고 예측
        - 모델이 탐지하지 않음   
    - Precision: 모델이 Object라 예측한 것 중에 실제 Object의 비율 
        - $ \text{Precision} = \frac{TP}{TP + FP} $
    - Recall: 실제 Object 중 모델이 예측하여 맞춘 Object의 비율
        - $\text{Recall} = \frac{TP}{TP + FN} $
    - Precision과 Recall의 조화 평균

### **Precision-Recall Curve**
![Precision-Recall Curve](https://github.com/user-attachments/assets/ff8c5308-8e54-4b02-8f44-9a433f59b58a)

Precision-Recall Curve는 Precision과 Recall을 모두 감안한 지표로 모델의 성능을 평가할 때 두 지표간의 관계를 시각적으로 표현
- AP
    - Average Precision는 Precision-Recall Curve 아래의 면적을 표현
    - 면적은 모델의 Precision과 Recall간의 trade-off를 반영
- mAP
    - mean Average Precision은 여러 클래스에 대한 Average Precision의 평균을 의미 
    - 다중 클래스 문제에서 전체 모델 성능을 종합적으로 평가하는데 사용 

## **YOLO**
이미지에서 객체를 실시간으로 탐지하는 능력<br>
이미지 전체를 한 번에 처리하여 객체의 클래스와 위치를 동시에 예측
- 버전
    - Yolov3: 3종류의 탐지 커널 사용하여 기존에 비해 Object의 크기 영향을 덜 받음
    - Yolov4: One-stage Detector에 대한 검증
    - Yolov5: PyTorch 기반으로 구현되어 경량화 및 사용자 친화성을 강조