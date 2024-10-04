--- 
title: "머신러닝 | Machine Learning" 
date: 2024-10-05 00:53:23 +0900
achieved: 2024-10-04 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Machine Learning]
---
---------- 	
> KT 에이블스쿨 6기 머신러닝에 진행한 강의 내용 정리한 글입니다. 
{: .prompt-info } 

## **학습 방법에 따른 분류**
- 지도학습(Supervised Learning): 학습 대상이 되는 데이터에 정답을 주어 규칙성, 즉 데이터의 패턴을 배우게 하는 학습 방법
- 비지도학습(Unsupervised Learning): 정답이 없는 데이터만으로 배우게 하는 학습 방법
- 강화학습(Reinforcement Learning): 선택한 결과에 대해 보상을 받아 행동을 개선하면서 배우게 하는 학습 방법

## **과제에 따른 분류**
- 분류 문제(Classification): 이미 적절히 분류된 데이터를 학습하여 분류 규칙을 찾고 그 규칙을 기반으로 새롭게 주어진 데이터를 적절히 분류하는 것을 목적으로 합니다. 이때 분류는 범주값을 예측합니다. (지도 학습)
- 회귀 문제(Regression): 이미 결과값이 있는 데이터를 학습하여 입력값과 결과값의 연관성을 찾고 그 연관성을 기반으로 새롭게 주어진 데이터에 대한 값을 예측하는 것을 목적으로 합니다. 이때 회귀는 연속적인 숫자를 예측합니다. (지도 학습)
- 클러스터링(Clustering): 주어진 데이터를 학습하여 적절한 분류 규칙을 찾아 데이터를 분류함을 목적으로 하며 정답이 없으니 성능 평가하기가 어렸습니다. (비지도 학습)

## **모델 & 모델링 & 오차**
- 모델(Model): 데이터로부터 패턴을 찾아 수학식으로 정리해 놓은 것
- 모델링(Modeling): 오차가 적은 모델을 만드는 과정
- 모델의 목적: 샘플(표본, 부분집합)을 가지고 전체(모집단, 전체 집단)을 추정
- 오차: 관측값(실제값)과 모델 예측값의 차이(이탈도, Deviance)

## **데이터 분리**
데이터셋을 학습용(Training data), 검증용(Validation data), 평가용 데이터(Testing data)로 분리

```python
# 학습용, 평가용 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 
```

## **과대적합 & 과소적합**
- 과대적합(Overfitting): 학습 데이터에 대해서는 성능이 매우 좋은데 평가 데이터에 대해서는 성능이 좋지 않은 경우
- 과소적합(Underfitting): 학습 데이터보다 평가 데이터에 대한 성능이 매우 좋거나 모든 데이터에 대한 성능이 매우 안 좋은 경우, 모델이 너무 단순하여 학습 데이터에 대해 적절히 훈련되지 않은 경우 

## **모델 평가**
### **분류 모델**
- 분류 모델은 0인지 1인지를 예측하는 것 
- 예측값이 실제값과 가까울수록 좋은 모델
- 정확히 예측한 비율로 모델 성능을 평가 

### **회귀 모델**
- 회귀 모델이 정확한 값을 예측하기는 사실상 어려움
- 예측값과 실제값에 차이(오차)가 존재할 것으로 예상
- 예측값이 실제값과 가까울수록 좋은 모델
- 예측값과 실제값의 차이(오차)로 모델 성능을 평가 

## **평가 지표**
### **회귀 모델 평가 지표**

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \quad \quad RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \quad \quad MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- MSE: 오차 제곱(SSE)의 합을 구한 후 평균을 구합니다.
- RMSE: 오차 제곱이므로 루트한 값이다.
- MAE: 오차 절대값의 합을 구한 후 평균을 구합니다. 
- MAPE: 오차 비율을 표시하고 싶은 경우 사용합니다.
- 결정계수 $ R^2 $: 전체 오차 중에서 회귀식이 잡아낸 오차 비율로 일반적으로 0 ~ 1 사이의 값을 가지며 오차의 비 또는 설명력이 부릅니다. 

→ 위 값 모두 작을수록 모델 성능이 좋습니다. <br>
→ $ R^2 $ = 1이면 MSE = 0 은 모델이 데이터를 완벽학 학습한 것입니다. 

### **분류 모델 평가 지표**

![오분류표](https://github.com/user-attachments/assets/4146678e-f777-4fba-b2eb-29700fde52c1)<p align="center">오분류표</p>

**오분류표**
- TN(True Negative, 진음성): 음성을 음성이라고 예측한 것
- FP(False Positive, 위양성): 음성을 양성이라고 예측한 것
- FN(False Negative, 위음성): 양성을 음성이라고 예측한 것
- TP(True Positive, 진양성): 양성을 양성이라고 예측한 것

**평가지표**

$$
\text{정확도 (Accuracy)} = \frac{TP + TN}{TP + TN + FP + FN} \quad \quad \quad \text{정밀도 (Precision)} = \frac{TP}{TP + FP}
$$

$$
\text{재현율 (Recall)} = \frac{TP}{TP + FN} \quad \quad \quad \text{특이도 (Specificity)} = \frac{TN}{TN + FP}
$$

$$
F1 \text{ Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$


- 정확도(Accuracy): 전체 중에서 Positive와 Negative 로 정확히 예측한(TN + TP) 비율
- 정밀도(Precision): Positive로 예측한 것(FP + TP) 중에서 실제 Positive(TP)인 비율
- 재현율(Recall): 실제 Positive(FN + TP) 중에서 Positive로 예측한(TP) 비율
- 특이도(Specificity): 실제 Negative(TN + FP) 중에서 Negative로 예측한(TN) 비율
- F1-Score: 정밀도와 재현율의 조화 평균

→ 정밀도와 재현율은 trade-off 관계

## **Linear Regression**
선형회귀: 함수 $y = ax + b$의 최적 기울기 $a$와 $y$ 절편 $b$를 결정하는 방법
- 최적의 회귀모델: 전체 데이터의 오차 합이 최소가 되는 모델을 의미
- 오차합이 최소가 되는 가중치 $w_1$과 편향 $w_0$를 결정

### **단순회귀**
- 독립변수 하나가 종속변수에 영향을 미치는 선형 회귀
- $x$값 하나만으로 $y$값을 설명할 수 있는 경우 
- 회귀식: $y = w_0 + w_1 x $

```python
# 회귀계수 확인
print(model.coef_)
print(model.intercept_)
```
### **다중회귀**
- 여러 독립변수가 종속변수에 영향을 미치는 선형 회귀
- $y$ 값을 설명하기 위해서는 여러 개의 $𝑥$ 값이 필요한 경우
- 회귀식: $y = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n$

```python
# 회귀계수 확인
print(list(x_train))
print(model.coef_)
print(model.intercept_)
```
### **회귀모델 구현**
Linear Regression 알고리즘은 회귀 모델에만 사용
- 알고리즘 함수: `sklearn.linear_model.LinearRegression`
- 성능평가 함수: `sklearn.metrics.mean_absolute_error, sklearn.metrics.r2_score`

```python
# 불러오기
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 선언하기
model = LinearRegression()

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```
## **K-Nearest Neighbor**
- 학습용 데이터에서 k개의 최근접 이웃의 값을 찾아 그 값들로 새로운 값을 예측하는 알고리즘
- k 값에 따라 예측 값이 달라지므로 적절한 k 값을 찾는 것이 중요
- 회귀와 분류에 사용되는 매우 간단한 지도학습 알고리즘

### **거리 구하는 방법**

$$\text{Euclidean Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \quad \quad \text{Manhattan Distance} = |x_2 - x_1| + |y_2 - y_1|$$

- 유클리드 거리(Euclidean Distance): 두 점 사이의 직선 거리
- 맨해튼 거리(Manhattan Distance): 두 점 사이를 수직 및 수평으로 이동할 때의 거리

### **스케일링(Scaling)**
스케일링 여부에 따라 모델 성능이 달라질 수 있습니다.

$$
X_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \quad \quad \quad X_{z} = \frac{x - \bar{x}}{s}
$$

- 정규화(Normalization): 각 변수의 값이 0과 1사이 값
- 표준화(Standardization): 각 변수의 평균이 0, 표준편차가 1이 됨

→ 평가용 데이터에도 학습용 데이터를 기준으로 스케일링을 수행합니다. 

```python
# 함수 불러오기
from sklearn.preprocessing import MinMaxScaler

# 정규화
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

### **회귀모델 구현**
- 알고리즘 함수: `sklearn.neighbors.KNeighborsRegressor`
- 성능평가 함수: `sklearn.metrics.mean_absolute_error, sklearn.metrics.r2_score`

```python
# 불러오기
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 선언하기
model = KNeighborsRegressor(n_neighbors=5)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```

### **분류모델 구현**
- 알고리즘 함수: `sklearn.neighbors.KNeighborsClassifier`
- 성능평가 함수: `sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report` 

```python
# 불러오기
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 선언하기
model = KNeighborsClassifier(n_neighbors=5)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## **Decision Tree**
- 결정 트리(의사 결정 나무)로 특정 변수에 대한 의사결정 규칙을 나무 가지가 뻗은 형태로 분류
- 분류와 회귀 모두에 사용되는 지도학습 알고리즘 

→ 과적합으로 모델 성능이 떨어지기 쉬움<br>
→ 트리 깊이를 제한하는(=가지치기) 튜닝이 필요

### **Decision Tree 용어**
- Root Node(뿌리 마디): 전체 자료를 갖는 시작하는 마디
- Child Node(자식 마디): 마디 하나로부터 분리된 2개 이상의 마디
- Parent Node(부모 마디): 주어진 마디의 상위 마디
- Terminal Node(끝 마디): 자식 마디가 없는 마디(=Leaf Node)
- Internal Node(중간 마디): 부모 마디와 자식 마디가 모두 있는 마디
- Branch(가지): 연결되어 있는 2개 이상의 마디 집합
- Depth(깊이): 뿌리 마디로부터 끝 마디까지 연결된 마디 개수

### **분류 모델 평가 지표**
**지니 불순도**

$$Gini(S) = \sum_{i=1}^{c} p_i(1 - p_i)
$$

지니 불순도 = 1 - ($양성 클래스 비율^2$ + $음성 클래스 비율^2$)
- 분류후에 얼마나 잘 분류했는지 평가하는 지표입니다. 
- 지니 불순도는 이진 분류의 경우 0 ~ 0.5 사이의 값을 가지며 불순도가 낮을수록 순도가 높습니다. 
- 지니 불순도가 낮은 속성으로 의사결정 트리 노드가 결정됩니다. 

**엔트로피(Entropy)**

$$Entropy = -\sum_{i=1}^{m} p_i \log_2 p_i$$


엔트로피 = − 음성클래스비율 × $𝑙𝑜𝑔_2$ (음성클래스비율) − 양성클래스비율 × $𝑙𝑜𝑔_2$(양성클래스비율)
- 엔트로피는 0 ~ 1 사이의 값을 가지며 완벽하게 분류되면 0을, 완벽하게 섞이면(50:50) 1을 가집니다. 

**정보 이득(Information Gain)**
$$𝐺𝑎𝑖𝑛 𝑇, 𝑋 = 𝐸𝑛𝑡𝑟𝑜𝑝𝑦 𝑇 − 𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝑇, 𝑋)$$

특정 속성으로 데이터를 분할했을 때 불순도가 얼마나 감소하는지를 측정하는 지표
- 정보 이득이 크다는 것은 어떤 속성으로 분할할 때 불순도가 줄어든다는 것을 의미합니다.
- 모든 속성에 대해 분할한 후 정보 이득을 계산합니다.
- 정보 이득이 가장 큰 속성부터 분할합니다.

### **가지치기**
- 가지치기를 하지 않으면 모델이 학습 데이터에는 매우 잘 맞지만 평가 데이터에 잘 맞지 않는 과대적합 혹은 일반화되지 못하는 모습을 보여줍니다. 
- 하이퍼 파라미터 값(max_depth, min_samples_leaf, min_samples_split 등)을 조정
- 학습 데이터에 대한 성능은 낮아지나 평가 데이터에 대한 성능을 높일 수 있습니다.

### 회귀모델 구현
- 알고리즘 함수: `sklearn.tree.DecisionTreeRegressor`
- 성능평가 함수: `sklearn.metrics.mean_absolute_error, sklearn.metrics.r2_score`

```python
# 불러오기
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 선언하기
model = DecisionTreeRegressor(max_depth=5)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```

### **분류모델 구현**
- 알고리즘 함수: `sklearn.tree.DecisionTreeClassifier`
- 성능평가 함수: `sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report`
```python
# 불러오기
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 선언하기
model = DecisionTreeClassifier(max_depth=5)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## **Logistic Regression**

$$p = \frac{1}{1 + e^{-f(x)}}
$$
- 시그모이드(sigmoid) 함수라고도 불리며 분류모델에만 사용할 수 있습니다. 
- 확률 값 p는 선형 판별식 값이 커지면 1, 작아지면 0에 가까운 값이 됩니다. 
- (-∞, ∞) 범위를 갖는 선형 판별식 결과로 (0, 1) 범위의 확률 값을 얻습니다. 

### **분류모델 구현**
- 알고리즘 함수: `sklearn.linear_model.LogisticRegression`
- 성능평가 함수: `sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report`

```python
# 불러오기
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 선언하기
model = LogisticRegression()

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## **K-Fold Cross Validation**
- 모든 데이터가 평가에 한번, 학습에 k-1번 사용됩니다. (k ≥ 2)
- K개의 분할(Fold)에 대한 성능을 예측한 후, 그 평균과 표준편차를 계산하여 모델의 일반화 성능을 평가합니다. 

### **장단점**
**장점**
- 모든 데이터를 학습과 평가에 사용할 수 있습니다.
- 반복 학습과 평가를 통해 정확도를 향상시킬 수 있습니다.
- 데이터가 부족해서 발생하는 과소적합 문제를 방지할 수 있습니다.
- 평가에 사용되는 데이터의 편향을 막을 수 있습니다. 
- 좀 더 일반화된 모델을 만들 수 있습니다. 

**단점**
- 반복 횟수가 많아서 모델 학습과 평가에 많은 시간이 소요됩니다. 

### **K-분할 교차 검증 사용** 

```pyhton
# 1단계: 불러오기
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 2단계: 선언하기
model = DecisionTreeClassifier(max_depth=3)

# 3단계: 검증하기
cv_score = cross_val_score(model, x_train, y_train, cv=10)

# 확인
print(cv_score)
print(cv_score.mean())
```

## **Hyperparameter**
알고리즘을 사용해 모델링할 때 모델 성능을 최적화하기 위해 조절할 수 있는 매개변수

### **KNN**
**k값**
- k값(n_neighbors)에 따라 성능이 달라집니다. 
- k 값이 가장 클 때(=전체 데이터 개수) 가장 단순 모델 → 평균, 최빈값
- k 값이 작을 수록 복잡한 모델이 됨

### **Decision Tree**
**max_depth**
- 트리의 최대 깊이 제한
- 이 값이 작을 수록 트리 깊이가 제한되어 모델이 단순해집니다.

**min_samples_leaf**
- leaf가 되기 위한 최소한의 샘플 데이터 수
- 이 값이 클수록 모델이 단순해집니다.

**min_samples_split**
- 노드를 분할하기 위한 최소한의 샘플 데이터 수
- 이 값이 클 수록 모델이 단순해집니다.

## **Grid Search & Random Search**
### **Random Search**
- 함수 불러오기: `RandomizedSearchCV`
- 파라미터 값 지정: 딕셔너리로 값 범위를 지정
- 모델 선언: n_iter에 수행 횟수 지정, 적절한 cv값 지정
- 모델 학습: 모델 학습 과정이 최선의 파라미터 값을 찾는 과정으로 경우에 따라 많은 시간이 소요될 수도 있습니다. 
- 결과 확인: 딕셔너리 형태로 결과

```python 
# 함수 불러오기
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

# 파라미터 선언
param = {'n_neighbors': range(1, 500, 10),
'metric': ['euclidean', 'manhattan']}

# 기본모델 선언
knn_model = KNeighborsClassifier()

# Random Search 선언
model = RandomizedSearchCV(knn_model,
param,
cv=3,
n_iter=20)

# 학습하기
model.fit(x_train, y_train)

# 수행 정보
model.cv_results_
# 최적 파라미터
model.best_params_
# 최고 성능
model.best_score_
```

### **Grid Search**
- 함수 불러오기: `GridSearchCV`
- n_iter 옵션을 지정하지 않습니다.
- 넓은 범위와 큰 Step으로 설정한 후 범위를 좁혀 나가는 방식으로 시간을 단축합니다.
```python
# 함수 불러오기
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 파라미터 선언
param = {'n_neighbors': range(1, 500, 10), 'metric': ['euclidean', 'manhattan']}

# 기본모델 선언
knn_model = KNeighborsClassifier()

# Grid Search 선언
model = GridSearchCV(knn_model, param, cv=3)
```

## **앙상블**
앙상블 방법에는 보팅(Voting), 배깅(Bagging), 부스팅(Boosting), 스태킹(Stacking) 4종류가 있습니다.

### **보팅(Voting)**
여러 모델들(다른 유형의 알고리즘 기반)의 예측 결과를 투표를 통해 최종 예측 결과를 결정하는 방법
- 하드 보팅: 다수 모델이 예측한 값이 최종 결과값
- 소프트 보팅: 모든 모델이 예측한 레이블 값의 결정 확률 평균을 구한 뒤 가장 확률이 높은 값을 최종 선택

### **배깅(Bagging)**
데이터로부터 부트스트랩 한 데이터로 모델들을 학습시킨 후, 모델들의 예측 결과를 집계해 최종 결과를 얻는 방법
- 범주형 데이터(Categorical Data): 투표 방식(Voting)으로 결과를 집계
- 연속형 데이터(Continuous Data): 평균으로 결과를 집계
- 대표적인 배깅 알고리즘: Random Forest

**랜덤 포레스트(Random Forest)**
배깅(Bagging)의 가장 대표적인 알고리즘<br>
여러 Decision Tree 모델이 전체 데이터에서 배깅 방식으로 각자의 데이터 샘플링<br>
모델들이 개별적으로 학습을 수행한 뒤 모든 결과를 집계하여 최종 결과 결정

### **부스팅(Boosting)**
이전 모델이 제대로 예측하지 못한 데이터에 대해서 가중치를 부여하여 다음 모델이 학습과 예측을 진행하는 방법
- 배깅에 비해 성능이 좋지만, 속도가 느리고 과적합 발생 가능성이 있습니다.
- 대표적인 부스팅 알고리즘: XGBoost, LightGBM
 
**XGBoost(eXtreme Gradient Boosting)**
부스팅을 구현한 대표적인 알고리즘 중 하나가 GBM(Gradient Boost Machine)<br>
회귀, 분류 문제를 모두 지원하며, 성능과 자원 효율이 좋아 많이 사용

## **스태킹(Stacking)**
여러 모델의 예측 값을 최종 모델의 학습 데이터로 사용하여 예측하는 방법
### **Random Forest - 회귀모델 구현**
- 알고리즘 함수: `sklearn.ensemble.RandomForestRegressor`
- 성능평가 함수: `sklearn.metrics.mean_absolute_error, sklearn.metrics.r2_score`

```python
# 불러오기
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 선언하기
model = RandomForestRegressor(max_depth=5, n_estimators=100, random_state=1)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```

### **Random Forest - 분류모델 구현**
- 알고리즘 함수: `sklearn.ensemble.RandomForestClassifier`
- 성능평가 함수: `sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report`
```python
# 불러오기
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 선언하기
model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=1)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
### **XGBoost - 회귀모델 구현**
- 알고리즘 함수: `xgboost.XGBRegressor`
- 성능평가 함수: `sklearn.metrics.mean_absolute_error, sklearn.metrics.r2_score`
```python 
# 불러오기
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 선언하기
model = XGBRegressor(max_depth=5, n_estimators=100, random_state=1)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```
### **XGBoost - 분류모델 구현**
- 알고리즘 함수: `xgboost.XGBClassifier`
- 성능평가 함수: `sklearn.metrics.confusion_matrix, sklearn.metrics.classification_report`

```python 
# 불러오기
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 선언하기
model = XGBClassifier(max_depth=5, n_estimators=100, random_state=1)

# 학습하기
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

# 평가하기
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```