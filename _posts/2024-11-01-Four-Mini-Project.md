--- 
title: "4차 미니 프로젝트 | Four Mini Project" 
date: 2024-11-03 11:10:45 +0900
achieved: 2024-11-01 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Visual Intelligence, Mini Project]
---
---------- 	
> KT 에이블스쿨 4차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
이미지 데이터 모델링 얼굴 인식 (Face Recognition)

## **데이터셋**
- 유명인 얼굴 데이터 및 개인 얼굴 데이터  
- [Face_recognition_1: Kaggle - Labelled Faces in the Wild (LFW) Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) 
- [Face_recognition_2: Roboflow - Dataset Versions](https://universe.roboflow.com/td-vgaen/test-uiodm/dataset/2)

## **가상환경 구축**

```
# 프로젝트 폴더 이동
cd C:\Users\User\KT_Project\M4

# 가상환경 생성
python -m venv proj4

# 가상환경 활성화
cd proj4\Scripts
activate

# 프로젝트 의존성 설치
cd .. 
cd .. 
pip install -r requirements.txt
```

## **데이터** 
### **데이터 수집**
**팀원 4명의 얼굴 사진** 
- 상반신 포함된 사진: 150장 
- 얼굴만 포함된 사진: 1050장 
<br>

**유명인 얼굴 사진(other_face)**
- Face_recognition_1
- Face_recognition_2

### **Annotation & Augmentation**
**Annotation**
- Roboflow 사용하여 팀원 5명의 얼굴 사진을 라벨링(Labeling)합니다.

**Augmentation**
- Rotatoin(회전): 이미지를 특정 각도인 -15° ~ +15° 사이의 임의 각도로 이미지를 회전합니다.
- Brightness(밝기): -15% ~ +15%의 밝기 변화를 조정합니다. 
- Blur(블러): 이미지를 흐리게 만들어 노이즈나 흐릿한 이미지에서 객체를 인식할 수 있도록 블러 적용합니다. 
- Noise(노이즈 추가): 이미지에 랜덤 노이즈를 추가하여 잡음을 시뮬레이션합니다. 
- Cutout(사진의 일부분 제거): 이미지의 일부를 무작위로 제거하여 해당 영역이 결여된 상태에서 객체를 인식하도록 합니다. 

## **Model**
### **Base Model**
사용 데이터셋: 팀원 5명의 얼굴 파일, Face_recognition_1, Face_recognition_2 <br>
사용 모델: Yolo11n

![Base Model](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img01.jpg?raw=true)
<p align="center">Base: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- 전반적으로 본인의 얼굴을 인식하지 못하고 other_face로 인식하는 경향이 보임
- 추가적인 개선이 필요

### **가설1. Train과정에서 데이터 증강**

![Train 증강](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img02.jpg?raw=true)
<p align="center">train 과정에서 증강하였을 때의 데이터 수</p>

Train 과정에서 한 번 데이터를 증강하기 
- Auto Augment: 미리 정의된 증강 정책을 자동으로 적용
- HSV Saturation: 이미지의 채도를 일부 변경
- Mosaic: 네 개의 훈련 이미지를 하나로 결합해 다양한 장면과 객체 상호작용을 시뮬레이션
- Mixup: 두 이미지와 해당 레이블을 혼합하여 합성 이미지 생성

![가설1 시각화](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img03.jpg?raw=true)
<p align="center">Train 과정에서 증강: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- Base Model에 비해 other_face로 탐지가 눈에 띄게 줄어듦
- 외국인 얼굴 위주의 Class인 other_face와 다르게 같은 한국인 Class로 분류됨
- 개선의 여지가 보임

### **가설2. Model 변경**

![yolo11m 사용](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img04.jpg?raw=true)
<p align="center">Yolo11m 사용하였을 때의 데이터 수</p>

Yolo11m 사용하기
- Base Model인 가장 가벼운 Yolo11n 대신 매개변수와 계산량이 조금 더 깊고 무거운 Yolo11m 사용

![가설2 시각화](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img05.jpg?raw=true)
<p align="center">Model 변경: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- 전체적으로 학습이 잘 된 모습
- 신뢰도(confidence score)는 상대적으로 낮지만 오탐률은 줄어듦
- 모델이 무거워서 실시간으로 객체 인식하는데 버퍼링이 생김

### **가설3. 데이터 전처리**

![가설3](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img06.jpg?raw=true)
<p align="center">other face인 데이터셋 Face_recognition_1, Face_recognition_2 시각화</p>

- Face_recognition_1: 각 이미지에 여러 개의 라벨이 많이 포함되어 있어 하나의 사진에 여러 사람이 존재할 가능성이 높은 데이터셋
- Face_recognition_2: 각 이미지에 한 개의 라벨이 많이 포함되어 있어 하나의 사진에 한 사람이 존재할 가능성이 높은 데이터셋 

#### **가설 3-1. Face_recognition_1만 사용** 
![가설3-1 데이터셋](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img07.jpg?raw=true)
<p align="center">Face_recognition_1만 사용하였을 때의 데이터 수</p>

- 하나의 이미지에 여러 개의 라벨이 포함된 데이터셋인 Face_recognition_1만 사용

![가설3-1 데이터셋](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img08.jpg?raw=true)
<p align="center">Face_recognition_1만 사용: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- other_face로 인식하는 경향이 크고 jaeyub이 아닌 다른 사람으로 인식함
- 우리의 목적과 맞지 않는 데이터셋

#### **가설 3-2-1. Face_recognition_2만 사용** 
![가설3-2-1 데이터셋](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img09.jpg?raw=true)
<p align="center">Face_recognition_2만 사용하였을 때의 데이터 수</p>

- 하나의 이미지에 한 개의 라벨이 많이 포함된 데이터셋인 Face_recognition_2만 사용

![가설3-2-1](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img10.jpg?raw=true)
<p align="center">Face_recognition_2만 사용: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- 오탐률이 줄어들고 신뢰도가 Base Model에 비해 월등히 올라감
- 우리의 목적에 부합하는 데이터셋

#### **가설 3-2-2. Face_recognition_2의 Label이 1인 값만 사용** 
![가설3-2-2 데이터셋](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img11.jpg?raw=true)
<p align="center">Face_recognition_2의 Label이 1인 값만 사용하였을 때의 데이터 수</p>

- Face_recognition_2 데이터셋에서 하나의 이미지에 한 개의 라벨 포함된 데이터셋 사용 

![가설3-2-1](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img12.jpg?raw=true)
<p align="center">Face_recognition_2의 Label이 1인 값만 사용: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- 모델 성능 향상에 미치는 영향이 거의 없음
- 1인 탐지를 목적으로 하기 때문에 가설 3-2-1의 모델보다 성능 향상을 기대하였지만 실패 

#### **가설 3-3. Face_recognition의 Class 랜덤 3000개 사용** 
![가설3-3 데이터셋](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img13.jpg?raw=true)
<p align="center">Face_recognition_2의 Label이 1인 값만 사용하였을 때의 데이터 수</p>

- 데이터의 불균형을 해소하기 위해 Face_recognition 데이터에서 무작위로 3000개를 선택하여 사용 

![가설3-2-1](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject4_img14.jpg?raw=true)
<p align="center">Face_recognition의 Class 랜덤 3000개 사용: 10초간 jaeyub이 움직이는 동안의 데이터 시각화</p>

- Class 불균형을 해결하기 위해 other face class의 데이터 수를 줄임
- other face로 오탐하는 비율은 줄었지만 jaeyub이 아닌 다른 Class로 오탐했음

## **결론 & 추가 성능 개선 방법**
**결론**<br>
데이터 수집 방식에 따라 성능 차이가 발생함을 확인했습니다. 또한, 목적에 맞는 학습 데이터를 선택하고 데이터 불균형을 고려하여 학습시킨 결과, 성능이 향상되었습니다.<br>
<br>

**추가 성능 개선 방법**

- 데이터의 다양성 유지
    - 다중인식인 경우, 같은 장소나 환경에서 촬영된 다양한 사진
- 클래스 불균형 해소 
    - 오버샘플링 혹은 언더샘플링을 통한 데이터 증강

## **고찰**
다양한 가설을 세워 직접 실험함으로써 데이터의 중요성을 다시 한번 깨달을 수 있는 계기가 되었습니다. <br>
데이터의 다양성을 확보하고 클래스 불균형 문제를 해결하는 것이 모델의 성능을 향상시키는 중요 요소라는 점을 확인할 수 있었습니다. <br>
다만, 시간이 부족하여 데이터셋 수집 및 다양한 모델을 실험을 충분히 진행하지 못한 점이 아쉬움으로 남습니다.  