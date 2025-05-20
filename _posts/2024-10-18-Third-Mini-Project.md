--- 
title: "3차 미니 프로젝트 | Third Mini Project" 
date: 2024-10-18 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Mini Project]
---
---------- 	
> KT 에이블스쿨 3차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
스마트폰 센서 기반 데이터를 활용한 행동 인식 

## **데이터셋**
- data01_test.csv, data01_train.csv, features설명.xlsx, features.csv 
- 모델 학습과 성능 평가에 사용되는 데이터 포함
- 모델이 행동을 인식하는 데 사용하는 입력 데이터 포함 

## **개인과제**
도메인 이해 및 데이터 전처리, 데이터 분석
### **데이터 분석** 
#### **모델링**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier()

# 모델 훈련
model.fit(x_train, y_train)

# 검증 세트에서 예측 수행
y_pred = model.predict(x_val)

# 검증 세트에서 성능 평가
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')
```
Validation Accuracy: 0.98의 결과로 높은 정확도를 확인할 수 있습니다. 

#### **변수 중요도**
![변수 중요도](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img01.jpg?raw=true)

목표 변수인 'Activity' 확인하기
- 데이터셋 내의 모든 클래스의 데이터 분포가 비교적으로 균일 
- 정적 행동(STANDING, LAYING, SITTING)가 동적 행동 (WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS)보다 데이터 개수가 더 많은 것을 확인 가능

#### **변수 중요도 추출**
![변수 중요도 추출](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img02.jpg?raw=true)

상위: angle(X,gravityMean), tGravityAcc-mean()-X, tGravityAcc-mean()-Y, tGravityAcc-max()-X, tGravityAcc-max()-Y<br>
하위: fBodyAcc-bandsEnergy()-57,64.1, tBodyAccJerk-mean()-Z, fBodyAccJerk-iqr()-Z, fBodyBodyAccJerkMag-entropy(), fBodyAccJerk-bandsEnergy()-33,40.1

#### **상위 변수 중요도** 
![상위 변수 중요도](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img03.jpg?raw=true)

LAYING 클래스는 해당 피처에서 명확하게 구분되는 패턴을 보이므로 해당 클래스를 분류하는데 중요한 피처일 가능성이 높다고 추정<br>
LAYING 상태일 때 신체의 앞뒤 방향(X축)이 중력과 평행하거나 거의 평행할 때(중력과 동일한 방향) 양의 값을 나타냅니다. 

- 상위 변수 5개
    - tGravityAcc-mean()-X: X축 방향 중력가속도 평균
    - tGravityAcc-mean()-Y: Y축 방향 중력가속도 평균
    - tGravityAcc-max()-X: X축 방향 중력가속도 최대값
    - tGravityAcc-max()-Y: Y축 방향 중력가속도 최대값

#### **하위 변수 중요도**
![하위 변수 중요도](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img04.jpg?raw=true)

정적 클래스와 동적 클래스에 속하는 데이터의 분포가 전체적으로 겹쳐있는 것으로 보아 분류에 유의미하지 않는 것으로 추정

- 하위 변수 5개
    - fBodyAcc-bandsEnergy()-57,64.1: 고속퓨리에변환을 거친 57~64 구간의 시간갭1에 대한  에너지 밴드 가속도
    - tBodyAccJerk-mean()-Z: Z축 방향 가속도변화비율 평균
    - fBodyAccJerk-iqr()-Z: 고속퓨리에변환을 거친 Z축 방향 가속도변화비율 3사분위수 - 1사분위수
    - fBodyBodyAccJerkMag: 유클리드 노름을 사용한 고속퓨리에변환을 거친 가속도변화비율 3차원 신호 크기 신호의 엔트로피
    - fBodyAccJerk-bandsEnergy()-33,40.1: 고속퓨리에변환을 거친 33~40 구간의 시간갭1에  대한 에너지 밴드 가속도 변화비율

### **정적/동적 행동으로 구분하여 분석**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Activity를 is_dynamic으로 변환
activity_mapping = {
    'STANDING': 0,
    'SITTING': 0,
    'LAYING': 0,
    'WALKING': 1,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 1
}

# Activity를 is_dynamic으로 변환하고 Activity 열 삭제
train_data['is_dynamic'] = train_data['Activity'].map(activity_mapping)
train_data.drop('Activity', axis=1, inplace=True)  # Activity 열 삭제

target = 'is_dynamic'
x = train_data.drop(target, axis=1)
y = train_data.loc[:, target]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)

# Create and train the model
model2 = RandomForestClassifier(random_state=42)
model2.fit(x_train, y_train)

# Make predictions
y_pred2 = model2.predict(x_val)

# Evaluate the model
print(classification_report(y_val, y_pred2))
print(f"Accuracy: {accuracy_score(y_val, y_pred2)}")
```
Accuracy: 0.9991503823279524의 결과로 높은 정확도를 확인할 수 있습니다. 

#### **상위 변수 중요도 추출**
![상위 변수 중요도 추출](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img05.jpg?raw=true)

- tBodyAccJerk-std()-X: X축 방향 가속도변화비율 표준편차
- fBodyAccJerk-bandsEnergy()-1,16: 고속퓨리에변환을 거친 1~16 구간의 에너지 밴드 가속도 변화비율
- fBodyAccJerk-max()-X: 고속퓨리에변환을 거친 X축 방향 가속도변화비율 최대값
- fBodyAccJerk-bandsEnergy()-1,24: 고속퓨리에변환을 거친 1~24 구간의 에너지 밴드 가속도 변화비율
- tBodyGyroJerk-iqr()-Z: Z축 방향 각속도변화속도 3사분위수 - 1사분위수

#### **상위 변수 중요도**
![상위 변수 중요도](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img06.jpg?raw=true)

- 분포가 확실히 구분되어 분류에 유의미한 것으로 추정
- 정적 데이터에 속한 데이터는 -1값에 좁게 분포되어 있으며, 동적 클래스에 속한 데이터는 넓게 분포되어 있습니다.
- 정적 클래스는 변화율이 거의 비슷하고 동적 클래스는 변화율이 불규칙합니다.

### **변수 그룹 중요도 분석**

![변수 그룹 중요도 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img07.jpg?raw=true)

- tBodyAccJerk가 주로 중요도가 높은 것을 확인할 수 있습니다. 

### **딥러닝 모델링**
```python
# 데이터 분할: x, y
target = 'Activity'

x = train_data.drop(target, axis = 1)
y = train_data.loc[:, target]

# 스케일링
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# y 레이블 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 분할: train, validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state = 20)

nfeatures = x_train.shape[1] #num of columns

# Sequential
model3 = Sequential( [Input(shape = (nfeatures,)),
                      Dense(128, activation = 'relu'),
                      Dense(64, activation = 'relu'),
                      Dense(32, activation = 'relu'),
                      Dense(16, activation = 'relu'),
                      Dense(6, activation = 'softmax')] )

model3.compile(optimizer=Adam(learning_rate=0.001), loss= 'sparse_categorical_crossentropy')

history3 = model3.fit(x_train, y_train, epochs = 100, validation_split=0.2).history

p3 = model3.predict(x_val)
p3 = p3.argmax(axis = 1)

print(confusion_matrix(y_val, p3))
print(classification_report(y_val, p3))
```
- validation 데이터에서 0.98의 정확도로 높은 성능을 보였습니다. 
- test 데이터에서 0.9748로 만든 모델 중 가장 높은 성능을 보였습니다. 

## **팀과제** 
개인과제의 데이터 분석을 바탕으로 팀원들과 모델링을 통한 스마트폰 센서 데이터 기반 모션 분류

### **Best Model**
![Best Model 구조](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img08.jpg?raw=true)

팀원들 각자가 만든 모델들 중에 test에서 최고의 모델을 비교하여 그 중에서 정확도가 가장 높게 나온 모델을 선정하였습니다. <br>
총 7개의 모델을 비교했고 그 중 평균 정확도가 0.98로 가장 높게 나온 다섯 번째 모델을 사용했습니다. 

- 각 입력은 특정 센서 데이터를 기반으로 18개의 Input() 레이어를 정의 
- 복잡한 모델 구조를 설계하기 위해 Functional API 사용 
- 각 입력 데이터에 대해 32개의 노드를 가진 Dense 레이어를 적용하여 초기 특징 추출
- 3개의 노드를 가진 Dense 레이어를 추가하여 중요한 특성만 선정
- 이후 개별적으로 처리된 각 센서 데이터를 하나의 통합된 특징 벡터로 결합한 후 병합된 데이터를 받아 다시 32개의 노드를 가진 Dense 레이어를 적용하여 학습

#### **1단계**
![Best Model 구조](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img09.jpg?raw=true)

정적 행동(LAYING, SITTING, STANDING)과 동적 행동(WALKING, WALKING-UP, WALKING-DOWN)을 구분하는 모델 생성 
- 정확도가 1로 높은 분류 결과를 보입니다. 

#### **2-1단계**

![정적 동작 세부 분류 모델](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img10.jpg?raw=true)

정적 행동(LAYING, SITTING, STANDING)를 분류하는 모델 생성<br>

- validation 데이터와 test 데이터 평가에서 0.97로 높은 정확도를 확인할 수 있었습니다. 

#### **2-2단계**

![동적 동작 세부 분류 모델](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject3_img11.jpg?raw=true)

동적 행동(WALKING, WALKING-UP, WALKING-DOWN)를 분류하는 모델 생성<br>

- validation 데이터와 test 데이터 평가에서 0.99로 높은 정확도를 확인할 수 있었습니다. 

## **고찰**
도메인 지식이 부족하여 변수 중요도를 추출하여 시각화하는 과정에서 이에 대한 의미를 파악하는데 어려움을 느꼈으나 팀원들과의 토론을 통해 다양한 해석이 나왔고 이를 통해 도메인 지식을 쉽게 이해할 수 있었습니다. <br> 
모델의 성능을 높이기 위해 다양한 방법을 시도하며 수업시간에 배운 내용과 그 외의 방법들을 직접 시도할 수 있는 좋은 기회가 될 수 있었습니다.<br>
팀원들이 각자가 시도한 방법들을 설명하고 의논하는 과정에서 서로 피드백을 주고받으며 더 나은 모델을 개발할 수 있었습니다. 그 외에도 다양한 층을 쌓은 모델에 대해서도 배울 수 있는 계기가 되었다고 생각합니다. 
