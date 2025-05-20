--- 
title: "5차 미니 프로젝트 | Five Mini Project" 
date: 2024-11-23 13:39:45 +0900
achieved: 2024-11-12 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, DataFrame, Processing, Deep Learning, Machine Learning, Mini Project]
---
---------- 	
> KT 에이블스쿨 5차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
KT AICE ASSO 시험 대비 

## **데이터셋**
다양한 데이터를 활용하여 실전과 같은 연습으로 AICE ASSO시험 대비 
- VOC 고객 해지 예측
- 고객 이탈 여부 예측
- 네비게이션 도착시간 예측

## **AICE 자격**
AICE는 인공지능 능력시험 (AI자격증)으로 인공지능 활용 능력을 평가하는 시험입니다.<br>
KT가 개발했고 한국경제와 함께 주관합니다.

### **AICE 자격 종류**
초등학생부터 성인까지, 비전공자부터 전문 개발자까지 생애주기별 필요한 AI 역량에 따라 5개의 level로 구성되어 있습니다. 
![image](https://github.com/user-attachments/assets/7b7cb0de-bc6b-48c8-b345-1420625e32b6)

#### **AICE Associate** 
AI 기술에 대한 기본 지식과 실무 적용 능력을 평가합니다. 이 자격은 AI 기초 학습자부터 AI를 활용한 비즈니스 혁신을 목표로 하는 실무자에게 적합하며, AI를 활용한 데이터 분석, 머신러닝, 딥러닝의 기본 개념과 응용 능력을 검증합니다.

## **VOC 고객 해지 예측**
고객의 VOC 정보를 바탕으로 해지 여부 예측하기

- 탐색적 데이터 분석: 필요한 라이브러리 설치 / Tabular 데이터 로딩 / 데이터의 구성 확인, 상관분석 / 데이터 시각화
- 데이터 전처리: 데이터 결측치 처리 / 라벨 인코딩, 원핫 인코딩 / x,y 데이터 분리 / 데이터 정규분포화, 표준화
- 머신러닝/딥러닝 모델링
- 모델 성능평가 및 그래프 출력

### **실행 코드** 
a. 필요한 라이브러리 설치

```python
# pip 이용해서 seaborn을 설치
!pip install seaborn

# numpy 별칭을 np로, pandas 별칭을 pd로 해서 임포트
import numpy as np
import pandas as pd

# matplotlib 라이브러리를 plt로, seaborn을 sns로 해서 임포트
import matplotlib.pyplot as plt
import seaborn as sns
```

b. 데이터 로딩

```python
# pandas read_csv 함수를 사용하여 voc_data.csv 파일을 읽어온 후 df에 저장
df = pd.read_csv('voc_data.csv')
```

c. 데이터 구성 확인

```python
# "df" DataFrame 이용해서 읽어들인 파일의 앞부분 5줄, 뒷부분 5줄을 출력
df.head(5) 
df.tail(5)

# 데이터프레임 정보(컬럼정보, Null 여부, 타입) 출력
df.info()

# 데이터프레임 인덱스 확인
df.index

# 데이터프레임 컬럼 확인
df.columns

# 데이터프레임 값(value)을 확인
df.values

# 데이터프레임의 계산 가능한 값들에 대한 통계치를 확인
df.describe()

# DataFrame 컬럼 항목에 Null 존재하는지 확인
df.isnull().sum()

# voc_trt_perd_itg_cd 컬럼의 데이터를 확인
df['voc_trt_perd_itg_cd']

# voc_trt_perd_itg_cd 컬럼 데이터별 건수를 나열
df['voc_trt_perd_itg_cd'].value_counts()
```

d. 데이터 결측치 처리

```python
# voc_trt_perd_itg_cd 컬럼에서 '_' 값이 차지하는 비율이 50%가 넘는 것을 확인하고, 이 voc_trt_perd_itg_cd 컬럼을 삭제
underscore_ratio = (df['voc_trt_perd_itg_cd'] == '_').mean()

if underscore_ratio > 0.5:
    df1 = df.drop(columns=['voc_trt_perd_itg_cd'])

# 'df1' DataFrame에서 '_' 값이 50% 이상되는 나머지 컬럼도 삭제
cols_to_drop = [col for col in df1.columns if (df1[col] == '_').mean() > 0.5]

df1 = df1.drop(columns=cols_to_drop)

# 'df1' DataFrame의 'cust_clas_itg_cd' 컬럼에 '_' 값이 몇 개 있는지 확인하여 출력
underscore_count = (df1['cust_clas_itg_cd'] == '_').sum()

print(underscore_count)

# df1의 남아있는 '_'값을 null로 변경: DataFrame replace 함수를 사용해서 모든 컬럼에 대해 '_'값을 null로 변경하고 df2에 저장
df2 = df1.replace('_', np.nan)

# df2의 컬럼별 Null 갯수를 확인
df2.isnull().sum()

# df2 데이터프레임 컬럼들의 데이터타입을 확인
df2.dtypes

# df2 데이터프레임에 대해 먼저, 'cust_clas_itg_cd' 컬럼의 최빈값을 확인하는 코드로 확인하고 다음으로, 이 컬럼의 Null 값을 최빈값으로 변경하세요(fillna 함수 사용). 처리된 데이터프레임은 df3에 저장
mode = df2['cust_clas_itg_cd'].mode()

df2['cust_clas_itg_cd'] = df2['cust_clas_itg_cd'].fillna(mode[0])
df3 = df2.copy()

# df3에 대해 'age_itg_cd'의 null 값을 중앙값(median)으로 변경하고 데이터 타입을 정수(int)로 변경하세요. 데이터 처리 후 데이터프레임을 df4에 저장
df3['age_itg_cd'] = pd.to_numeric(df3['age_itg_cd'], errors='coerce')

median_age = df3['age_itg_cd'].median()
df3['age_itg_cd'].fillna(median_age, inplace=True)
df3['age_itg_cd'] = df3['age_itg_cd'].astype(int)

df4 = df3.copy()

# df4에 대해 'cont_sttus_itg_cd'의 null 값을 최빈값(mode)으로 변경하세요. 데이터 처리 후 데이터프레임을 df5에 저장
mode = df4['cont_sttus_itg_cd'].mode()
df4['cont_sttus_itg_cd'] = df4['cont_sttus_itg_cd'].fillna(mode)
df5 = df4[:]

# df5에 대해 'cust_dtl_ctg_itg_cd'의 null 값을 최빈값(mode)으로 변경
mode = df5['cust_clas_itg_cd'].mode()
df5['cust_dtl_ctg_itg_cd'] = df5['cust_dtl_ctg_itg_cd'].fillna(mode)

# df5에 대해 다음 날짜 관련 컬럼을 확인 후 삭제 (날짜 관련 컬럼: new_date, opn_nfl_chg_date, cont_fns_pam_date)
date_cols = ['new_date', 'opn_nfl_chg_date', 'cont_fns_pam_date']
print(df5[date_cols])
df5.drop(columns=date_cols, axis=1, inplace=True)

# df5에 대해 'voc_mis_pbls_yn' 컬럼을 삭제
df5.drop('voc_mis_pbls_yn', axis=1, inplace=True)
```

e. 라벨 인코딩, 원핫 인코딩

```python
# df5에 대해 object 타입 컬럼을 cat_cols에 저장하세요. 그 중 cat_cols의 cust_clas_itg_cd 컬럼에 대해 LabelEncoder를 적용 (적용 후 df5에 저장)
from sklearn.preprocessing import LabelEncoder

cat_cols = list(df5.select_dtypes('object').columns)

encoder = LabelEncoder()
df5['cust_clas_itg_cd'] = encoder.fit_transform(df5['cust_clas_itg_cd'])

# df5의 나머지 object 컬럼에 대해서 One-Hot Encoding될수 있도록 Pandas의 get_dummies 함수를 적용 (적용 후 df6에 저장)
df6 = pd.get_dummies(df5, drop_first=True)
```

f. x, y 데이터 분리

```python
# df6에 대해 X, y 값을 가지고 8:2 비율로 Train , Test Dataset으로 나누기 (y 클래스 비율에 맞게 분리, y 값은 'trm_yn_Y' 컬럼, random_state는 42)
from sklearn.model_selection import train_test_split

target = 'trm_yn_Y'
y = df6[target]
X = df6.drop(target, axis=1)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=.8, stratify=y, random_state=42)
```

g. 데이터 정규분포화, 표준화

```python
# 사이킷런의 StandardScaler로 훈련데이터셋은 정규분포화(fit_transform)하고 테스트 데이터셋은 표준화(transform)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_s = scaler.fit_transform(train_X)
test_X_s = scaler.transform(test_X)
```

h. 머신러닝 모델링 & 모델 성능평가 및 그래프 출력

```python
# LogisticRegression 모델을 만들고 학습을 진행 (단, 규제강도C는 10으로 설정, 계산에 사용할 작업수 max_iter는 2000으로 설정하세요)
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(C=10, max_iter=2000)
model_lr.fit(train_X_s, train_y)

# 위 모델의 성능을 평가
# y값을 예측하여 confusion matrix를 구하고 heatmap 그래프로 시각화
# Scikit-learn의 classification_report를 활용하여 성능을 출력
from sklearn.metrics import confusion_matrix, classification_report

pred_y = model_lr.predict(test_X_s)
sns.heatmap(confusion_matrix(pred_y, test_y), annot=True)
plt.show()

print(classification_report(pred_y, test_y))

# DecisionTree 모델을 만들고 학습을 진행 (단, max_depth는 10, random_state는 42로 설정)
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
model_dt.fit(train_X_s, train_y)

# RandomForest 모델을 만들고 학습을 진행 (단, n_estimators=100, random_state=42 설정)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(train_X_s, train_y)

# XGBoost 모델을 만들고 학습을 진행 (단, n_estimators=5 설정)
from xgboost import XGBClassifier

model_xgb = XGBClassifier(n_estimators=5)
model_xgb.fit(train_X_s, train_y)

# Light GBM 모델을 만들고 학습을 진행 (단, n_estimators=3 설정)
from lightgbm import LGBMClassifier

model_lgbm = LGBMClassifier(n_estimators=3)
model_lgbm.fit(train_X_s, train_y)

# Linear Regression 모델을 연습으로 만들고 학습을 진행
x_data = np.array([1.6, 2.3, 3.5, 4.6]).reshape(-1,1)
y_data = np.array([3.3, 5.5, 7.2, 9.9])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_data, y_data)
```

i. 딥러닝 모델링 & 모델 성능 평가 및 그래프 출력

```python
# 해지여부를 분류하는 딥러닝 모델
import tensorflow as tf

il = tf.keras.layers.Input(shape=train_X.shape[1:])
hl = tf.keras.layers.Dense(64, activation='relu')(il)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

hl = tf.keras.layers.Dense(32, activation='relu')(hl)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

hl = tf.keras.layers.Dense(16, activation='relu')(hl)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

ol = tf.keras.layers.Dense(1, activation='sigmoid')(hl)

model = tf.keras.Model(il, ol)

model.summary()


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

model.compile(optimizer='adam', loss=tf.keras.metrics.binary_crossentropy, metrics=['acc'])
history = model.fit(train_X_s, train_y,
                    epochs=10, batch_size=10,
                    validation_data=(test_X_s, test_y),
                    callbacks=[es, mc])

# y_train, y_test를 원핫 인코딩 후 다중 분류하는 딥러닝 모델
from tensorflow.keras.utils import to_categorical

one_train_y = to_categorical(train_y)
one_test_y = to_categorical(test_y)

il = tf.keras.layers.Input(shape=train_X.shape[1:])
hl = tf.keras.layers.Dense(64, activation='relu')(il)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

hl = tf.keras.layers.Dense(32, activation='relu')(hl)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

hl = tf.keras.layers.Dense(16, activation='relu')(hl)
hl = tf.keras.layers.Dropout(.2)(hl)
hl = tf.keras.layers.BatchNormalization()(hl)

ol = tf.keras.layers.Dense(2, activation='softmax')(hl)

model = tf.keras.Model(il, ol)

model.summary()


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

model.compile(optimizer='adam', loss=tf.keras.metrics.categorical_crossentropy, metrics=['acc'])
history = model.fit(train_X_s, one_train_y,
                    epochs=10, batch_size=10,
                    validation_data=(test_X_s, one_test_y),
                    callbacks=[es, mc])

# 모델 성능을 평가해서 그래프로 표현
# 학습 정확도와 검증정확도를 그래프로 표시하고 xlabel에는 Epochs, ylabel에는 Accuracy, 범례에는 Train과 Validation으로 표시
result_df = pd.DataFrame(history.history)[['acc', 'val_acc']]
result_df.columns = ['Train', 'Validation']
result_df.plot.line()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 모델 성능을 평가해서 그래프로 표현
# 학습 손실과 검증 손실을 그래프로 표시하고 xlabel에는 Epochs, ylabel에는 Loss, 범례에는 Train Loss와 Validation Loss로 표시
result_df = pd.DataFrame(history.history)[['loss', 'val_loss']]
result_df.columns = ['Train Loss', 'Validation Loss']
result_df.plot.line()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# y값을 예측하여 y_test_pred에 저장하고 정확도를 출력
from sklearn.metrics import accuracy_score

pred_y = model.predict(test_X_s)
accuracy_score(np.argmax(pred_y, axis=1), test_y)
``` 

## **고객 이탈여부 예측**
통신 상품 이용정보를 바탕으로 고객의 이탈 여부 예측

- 탐색적 데이터 분석: 필요한 라이브러리 설치 / Tabular 데이터 로딩 / 데이터의 구성 확인, 상관분석 / 데이터 시각화
- 데이터 전처리: 데이터 결측치 처리 / 라벨 인코딩, 원핫 인코딩 / x,y 데이터 분리 / 데이터 정규분포화, 표준화
- 머신러닝/딥러닝 모델링
- 모델 성능평가 및 그래프 출력

### **실행 코드**
a. 필요한 라이브러리 설치

b. 데이터 로딩

c. df에서 불필요한 customerID 컬럼 삭제하고 df1에 저장

d. df1의 TotalCharges 컬럼의 타입을 float로 변경

```python
# TotalCharge의 컬럼 타입을 확인하는 코드를 작성
# ' ' 값을 0으로 변환하고 컬럼 타입을 float로 변경
# 전처리 후 데이터를 df2에 저장
df1['TotalCharges'].dtype
df1['TotalCharges'].replace([' '], ['0'], inplace=True)
df1['TotalCharges'] = df1['TotalCharges'].astype(float)
df2=df1.copy()
```

e. df2에서 churn 컬럼의 데이터별 개수를 확인하는 코드를 작성하고 Yes, No를 각각 1, 0으로 변환한 후 df3에 저장

f. df3의 모든 컬럼에 대해 결측치를 확인하는 코드를 작성하고 결측치를 처리

g. df4에서 SeniorCitizen 컬럼을 bar 차트로 확인해보고 불균형을 확인

```python
# df4에서 SeniorCitizen 컬럼을 bar 차트로 확인해보고 불균형을 확인
# SeniorCitizen 컬럼은 불균형이 심하므로 삭제
df4['SeniorCitizen'].value_counts().plot(kind='bar')
df4.drop('SeniorCitizen', axis=1, inplace=True)
df4.info()
```

h. df4에서 다음의 가이드에 따라 데이터를 시각화

```python
# tenure (서비스 사용기간)에 대해 히스토그램으로 시각화 
# tenure를 x 값으로 churn을 hue 값으로 사용하여 kdeplot으로 시각화 하고 '서비스 사용기간이 길어질 수록 이탈이 적다'에 대해 'O'인지 'X'인지 출력
# 'tenure','MonthlyCharges','TotalCharges' 컬럼간의 상관관계를 확인하여 heatmap으로 시각화하고 가장 높은 상관계수 값을 출력

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df4, x='tenure')
plt.show()

sns.kdeplot(data=df4, x='tenure', hue='Churn')
plt.show()
print('O')

sns.heatmap(df4[['tenure','MonthlyCharges','TotalCharges']].corr(), annot=True)
print(0.83)
```

i. df4에서 컬럼의 데이터 타입이 object인 컬럼들을 원-핫 인코딩

j. df5에 대해 Scikit-learn의 train_test_split 함수로 훈련, 검증 데이터를 분리

k. MinMaxScaler 함수를 'scaler'로 정의하고 데이터를 정규화

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
```

l. 머신러닝 모델링 & 모델 성능평가 및 그래프 출력

m. 딥러닝 모델링 & 모델 성능평가 및 그래프 출력

## **네비게이션 도착시간 예측**
네비게이션 데이터를 활용한 도착 시간 예측

- 탐색적 데이터 분석: 필요한 라이브러리 설치 / Tabular 데이터 로딩 / 데이터의 구성 확인, 상관분석 / 데이터 시각화
- 데이터 전처리: 데이터 결측치 처리 / 라벨 인코딩, 원핫 인코딩 / x,y 데이터 분리 / 데이터 정규분포화, 표준화
- 머신러닝/딥러닝 모델링
- 모델 성능평가 및 그래프 출력

### **실행 코드** 
a. 필요한 라이브러리 설치

b. 데이터 로딩

c. Address1(주소1)에 대한 분포도 확인

```python
# Seaborn을 활용
# Address1(주소1)에 대해서 분포를 보여주는 countplot그래프 시각화
# 지역명이 없는 '-'에 해당되는 row(행)을 삭제
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x = 'Address1', data = df)
plt.show()

df.drop(df[df['Address1'] == '-'].index, inplace = True)
```

d. 실주행시간과 평균시속의 분포 확인

```python 
# Seaborn을 활용
# X축에는 Time_Driving(실주행시간)을 표시하고 Y축에는 Speed_Per_Hour(평균시속)을 표시

sns.jointplot(x = "Time_Driving", y = "Speed_Per_Hour", data = df)
plt.show()
```

e. 위의 jointplot 그래프에서 시속 300이 넘는 이상치를 발견, jointplot 그래프에서 발견한 이상치 1개를 삭제

f. 모델링 성능을 제대로 얻기 위해서 결측치 처리는 필수, 결측치 처리

g. 모델링 성능을 제대로 얻기 위해서 불필요한 변수는 삭제, 불필요 데이터 삭제 처리 

h. 원-핫 인코딩(One-hot encoding)은 범주형 변수를 1과 0의 이진형 벡터로 변환하기 위하여 사용하는 방법으로 원-핫 인코딩으로 조건에 해당하는 컬럼 데이터 변환

i. 훈련과 검증 각각에 사용할 데이터셋을 분리

```python
# Time_Driving(실주행시간) 컬럼을 label값 y로, 나머지 컬럼을 feature값 X로 할당한 후 훈련데이터셋과 검증데이터셋으로 분리
# 대상 데이터프레임: df_preset
# 훈련 데이터셋 label: y_train, 훈련 데이터셋 Feature: X_train
# 검증 데이터셋 label: y_valid, 검증 데이터셋 Feature: X_valid
# 훈련 데이터셋과 검증데이터셋 비율은 80:20, random_state: 42
# Scikit-learn의 train_test_split 함수를 활용
from sklearn.model_selection import train_test_split
x = df_preset.drop(columns = ['Time_Driving'])
y = df_preset['Time_Driving']
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 42)
```

j. 머신러닝 모델링 & 모델 성능평가 및 그래프 출력

k. 딥러닝 모델링 & 모델 성능평가 및 그래프 출력

## **Tip**
- 시험시간은 90분이지만 1시간 내로 완료할 수 있도록 연습해두기 
    - AICE 홈페이지에서 제공하는 샘플 여러번 응시
- 검색 및 블로그 참고 가능 (GPT 사용 불가)
    - 인터넷 검색이 가능하므로 아예 모르는 내용이 나오면 빠른 시간 내에 해결하는 연습 필요
- 머신러닝 및 딥러닝 부분의 배점이 크기 때문에 확실하게 외워두기
    - 성능을 높이거나 수준 높은 코드를 작성하였는지는 가점 여부 X 

![image](https://github.com/user-attachments/assets/bf23e842-ecf9-4c9c-a1d9-8e8cf9bdd15d)
![image](https://github.com/user-attachments/assets/66f7a3e0-95c6-4bca-88df-0f2aa9ff0d96)


자세한 코드는 [Github](https://github.com/tae2on/KT_aivle_school_AI_track/tree/main/MiniProject05)
에서 확인할 수 있습니다.
