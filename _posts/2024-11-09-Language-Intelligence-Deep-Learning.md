---
title: "언어지능 딥러닝 | Language Intelligence Deep Learning" 
date: 2024-11-09 23:29:24 +0900
achieved: 2024-11-08 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Language Intelligence]
---
----------
> KT 에이블스쿨 6기 언어지능 딥러닝에 진행한 강의 내용 정리한 글입니다. 
{: .prompt-info } 

## **TF-IDF**
자연어 처리에서 텍스트의 중요성을 평가하는 중요한 방법 

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

$t$:특정 단어, $d$: 특정 문서, $D$: 전체 문서 집합

- TF: 특정 단어가 문서에서 얼마나 자주 나타나는지를 나타내는 지표
- IDF: 특정 단어가 여러 문서에서 얼마나 일반적인지를 측정

### **단어 표현**
벡터는 언어적인 특성을 반영하여 단어를 수치화하는 방법입니다. 

#### **One-Hot Encoding**
단어를 0과 1로 이루어진 벡터로 표현하는 방식 

- 자연어 단어 표현에는 부적합 
    - 단어의 의미나 특성을 표현할 수 없음
    - 단어의 수가 매우 많으므로 고차원 저밀도 벡터를 구성함

#### **분포가설 기반**
같은 문맥의 단어, 즉 비슷한 위치에 나오는 단어는 비슷한 의미

- 카운트 기반 방법(Count-based): 특정 문맥 안에서 단어들이 동시에 등장하는 횟수를 직접 셈
- 예측 방법(Predictive): 신경망 등을 통해 문맥 안의 단어들을 예측

### **유사도** 
텍스트가 얼마나 유사한지를 표현하는 방식

- 유사도를 측정하기 전에 단어를 벡터화하여야 함(TF-IDF 활용)
    - TF-IDF로 벡터화한 값은 자카드 유사도를 제외한 모든 유사도 판단에서 사용
    - 자카드 유사도는 단어의 교집합과 합집합을 이용하여 벡터화 없이 단어 집합 간 유사도를 측정 할 수 있음 

#### **자카드 유사도**
두 무장을 각각 단어의 집합으로 만든 뒤 집합을 통해 유사도 측정

- 유사도 측정방법
    - A: 두 집합의 교집합인 공통된 단어의 개수
    - B: 집합이 가지는 단어의 개수 
- 0에서 1 사이의 값을 가짐

#### **코사인 유사도**
두 개의 벡터값에서 코사인 각도를 구하는 방법 

- -1에서 1사이의 값을 가짐

#### **유클리디언 유사도**
두 벡터 간의 거리를 유사도를 판단(기준: 유클리디언 거리판단)

#### **맨하튼 유사도**
두 벡터 가의 거리로 유사도를 판단 (기준: 맨하튼 거리판단)

### **거리 측정 방식**
#### **민코프스키 거리** 
두 점 사이의 거리(유사도)를 측정하는 일반화된 방식으로 다양한 값의 p에 따라 다른 거리 측정 방법을 표현

$$
d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

- $p$ = 1: Manhattan Distance(맨하튼 거리)
    - 두 벡터 사이의 축을 따라 이동한 거리의 합을 계산하는 방법
- $p$ = 2: Euclidean Distance(유크리드 거리)
    - 일반적인 직선 거리를 나타내며 피타고라스 정리를 기반으로 계산 

#### **코사인** 
벡터 $\vec{x}$와 벡터 $\vec{y}$ 사이의 각도 차이를 기반으로 유사성을 측정하는 방법 

$$
\text{cos}(\vec{x}, \vec{y}) = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \cdot \sqrt{\sum_{i=1}^{n} y_i^2}}
$$

- 코사인 거리는 코사인 유사도를 1에서 뺀 값으로 값의 범위는 $0 \leq d(\vec{x}, \vec{y}) \leq 2$

## **비지도학습**
### **분류(Classification)**
- 지도 학습(Supervised Learning): 분류는 주어진 데이터에 대해 레이블이 있는 인스턴스를 통해 학습
- 예측 방법 학습: 분류 모델은 이전에 분류된 인스턴스를 기반으로 새로운 인스턴스의 클래스를 예측하는 방법을 학습

### **군집화(Clustering)**
- 비지도 학습(Unsupervised Learning): 군집화는 주어진 데이터에 레이블이 없는 경우, 유사한 데이터끼리 그룹화하여 패턴을 찾아내는 방법
- 데이터 구조 파악: 군집화 모델은 데이터의 분포나 숨겨진 그룹을 찾기 위해 유사성을 기준으로 데이터를 군집

## **데이터 유형 및 표현 방식**
### **데이터 유형**
- 이산형 
    - 유한한 값의 집합으로 이루어진 특징 
    - 보통 정수형 변수로 표현
- 연속형 
    - 실수 값을 특징으로 가짐
    - 부동 소수점(float) 변수로 표현

### **데이터 표현**
![img](https://github.com/user-attachments/assets/47c07096-a5a9-4b69-9e92-15fa08cdc252)

- 데이터 행렬(Data Matrix)
    - 객체(관측값)-특징 구조로 데이터 포인트가 행, 특징이 열에 배치

![거리](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/Distance_Matrix.png?raw=true)


- 거리(비유사성) 행렬(Distance/Dissimilarity Matrix)
    - 두 데이터 포인트 사이의 거리를 나타내며 대칭적 또는 삼각형 모양의 행렬로 나타냄
    - 거리 행렬에서는 n개의 데이터 포인트의 각 쌍에 대한 거리가 계산되어 같은 데이터는 거리가 0임

## **Partitioning Approach**
데이터를 k개의 클러스터로 분할하는 방식으로 주어진 데이터를 여러 군집으로 나누어 각각의 군집에 속하는 데이터 간의 유사성을 최대화하고 다른 군집과의 차이를 최대화하는 것을 목표로 하며, k의 수를 미리 지정해줘야 합니다.
- K-means   
    - 데이터를 k개의 클러스터로 나누고 각 클러스터의 중심과 가장 가까운 데이터 포인트를 반복적으로 할당하여 최적의 클러스터 구조를 찾는 방법 
- K-medoids
    - K-means와 유사한 방법으로 각 클러스터의 중심을 실제 데이터 포인트 중 선택하여 이상치에 강건한 클러스터링 방법 
- CLARANS
    - K-medoids의 개선된 방식으로 랜덤 서치를 통해 대규모 데이터에서 효율적으로 클러스터 중심을 찾는 방법 
- KNN 
    - 새로운 데이터 포인트가 주어졌을 때 가장 가까운 k개의 이웃 데이터 포인트를 기준으로 분류하거나 회귀를 수행하는 방식
    - 주로 비지도 학습보다는 지도 학습에서 주로 사용되며 거리 측정을 통해 데이터를 분류


![knn](https://github.com/user-attachments/assets/a57451d9-eee5-4b00-9e46-0e48804ac93c)

## **Word Embedding**
자연어 처리에서 단어를 고정된 크기의 실수 벡터로 변환하는 방법으로 기계가 단어의 의미를 수치적으로 이해할 수 있도록 도와줍니다. 분포가설을 이용하여 단어를 조합시킵니다. 

### **Word2Vec**
단어 간의 의미적 유사성을 주변 단어의 맥락을 통해 학습합니다. 단어가 등장하는 문맥을 기반으로 단어 간의 관계를 파악하고 비슷한 맥락에서 자주 등장하는 단어들이 벡터 공간에서 가깝게 배치되도록 합니다. 

## **정보검색 & 추천 시스템** 
### **파레토 법칙 & 롱테일 법칙**
- 파레토 법칙: 상위 20%가 80%의 가치를 창출한다. 
- 롱테일 법칙: 하위 80%의 다수가 상위 20%보다 뛰어난 가치를 창출한다. 

### **정의 및 개요**
**정의**<br>
사용자의 행동 이력, 사용자 간 관계, 상품 유사도, 사용자 컨텍스트에 기반하여 사용자의 관심 상품을 자동으로 예측하고 제공하는 시스템 

**개요**
- CF(Collaborative Filtering)
    - 고객의 행동 이력을 기반으로 고객의 소비 패턴을 마이닝, 고객-고객, 아이템-아이템, 고객-아이템간 유사도를 측정, 유사도에 기반하여 아이템을 추천하는 방식
    - Item-to-Item CF
        - 사용자의 구매/방문/클릭 이력에 의존한 추천으로 Cosine Similarity 사용 
        - 피 구매 기록을 바탕으로 모든 아이템 쌍 사이의 유사도를 구하고 사용자가 구매한 아이템들을 바탕으로 다른 아이템을 추천 
        
**장단점**
- 장점
    - 최소한의 기본 정보만으로도 구현 가능 
    - 다양한 적용사례에서 적절한 정확도를 보장
- 단점
    - 고차원 저밀도 Vector Sparseness Issue
    - 새로운 사용자나 아이템이 추가되는데 따르는 확장성(Scalability)이 떨어짐 

### **알고리즘 개요** 
- Collaborative Filtering(CF)
    - 기존 아이템 간 유사성을 단순하게 비교하는 것에서 벗어나 데이터 안에 내재한 패턴을 이용하는 기법
        - 데이터에 내재되어 있는 패턴/속성을 알아내는 것이 핵심 기술
        - LSA 사용
        - SVD등의 기법을 사용하여 User와 Item을 동일한 차원의 잠재 속성 공간으로 투사, 차원축소를 통해 자료부족과 확장성의 문제를 해소하고 예측의 적중율을 높임
        - 새로 추가된 아이템도 추천 가능
- Content-based Filtering (CBF)
    - 아이템의 속성에 기반하여 유사 속성 아이템을 추천
    - 협업 필터링이 사용자의 행동이력을 이용하는 반면, 콘텐츠 기반 필터링은 아이템 자체를 분석하여 추천을 구현
        - 아이템의 내용을 분석해야 하므로 아이템 분석 및 유사도 측정이 핵심 
        - 자연어처리와 정보검색(TF-IDF)의 기술 사용
        - 잠재적인 특징들을 고려, 보다 다양한 범위의 추천 가능 

→ 유사성, 잠재요소 등을 고려하여 CBF, CF 알고리즘과 딥러닝 특징을 개발

## **언어지능 인공신경망** 
### **Linear Functions**
#### **Linear Regression**
**선형 모델**

$$
H(x) = W x + b
$$

$H(x)$: 예측된 값, $W$: 가중치, $b$: 편향, $x$: 입력 데이터 

**비용함수**

주어진 $x$와 $y$에 대해 모델의 예측값과 실제값 간의 차이를 계산하는 함수이다. 

$$
\text{Cost}(W, b) = \frac{1}{m} \sum_{i=1}^{m} \left( H(x_i) - y_i \right)^2
$$

$m$: 훈련 데이터의 수, $H(x_i)$: 모델의 예측값, $y_i$: 실제값

**경사하강법**

비용 함수를 최소화하는 방법으로 각 피라미터에 대해 미분을 계산하고 그 값을 바탕으로 가중치를 갱신

$$
W \leftarrow W - \alpha \frac{\partial \text{Cost}(W)}{\partial W}
$$

$alpha$: 학습률(Learning rate)로 가중치 갱신 속도를 조절하는 하이퍼파라미터

#### **Binary Classification**

**로지스틱/시그모이드 함수**

$$
g(X) = \frac{1}{1 + e^{-W^T X}}
$$

이진분류에서 사용

**비용함수**

$$
g(z) = \frac{1}{1 + e^{-W^T X}}
$$

#### **Softmax Classification**

**각 클래스에 대한 확률값 계산**

$$x_{1} W_{A1} + x_{2} W_{A2}$$

$$x_{1} W_{B1} + x_{2} W_{B2}$$

$$x_{1} W_{C1} + x_{2} W_{C2}$$

다중분류 문제에 사용 <br>
입력이 ($x_{1}, x_{2}$)에 대해 세 개의 클래스 A,B,C에 대한 예측을 수행합니다. 각 클래스에 대해 확률 값을 계산하여 가장 높은 확률을 가진 클래스를 선택하는 방식입니다. 

**비용함수**

$$
C(S, L) = - \sum L_i \log(S_i)
$$

$L$: 예측값, $S$: 실제값

### **Nonlinear Functions**
#### **ANN**
여러 개의 노드(뉴런)가 연결되어 있는 구조로, 입력 데이터와 출력 결과 사이의 비선형 관계를 모델링하는데 사용

- Relu함수
    - ReLU 함수는 입력이 0보다 작으면 0을 출력하고 0보다 크면 그 값을 그대로 출력
- CNN 
    - 이미지를 처리할 때 중요한 특징을 자동으로 추출
    - CNN for CIFAR-10
        - 10개의 클래스로 구성된 60,000개의 32x32 크기의 컬러 이미지로 이루어진 데이터셋으로 이미지 분류 문제를 다루는데 사용

### **Advanced Topics**
#### **GAN**
생성 모델의 일종으로 두 개의 신경망(생성자, 구분자)이 서로 경쟁하는 방식으로 학습하는 모델
- Generator(생성자)
    - Discriminator(구분자)가 생성한 이미지를 진짜 이미지처럼 인식하게 만드는 것 
    - 주어진 랜덤 노이즈 벡터를 받아들여 가짜 이미지를 생성
    - 학습을 통해 점점 더 진짜와 비슷한 이미지를 생성
- Discriminator(구분자)
    - 진짜 이미지와 가짜 이미지를 구분하는 역할
    - 생성자가 생성한 이미지를 진짜 이미지와 구분하려고 하며, 진짜 이미지와 가짜 이미지의 차이를 구별하는 방식으로 학습 

#### **Interpolation**
두 점 사이 또는 여러 점 사이의 값을 추정하는 수학적 기법<br>
주로 연속적인 데이터를 다룰 때 유용하며 다양한 분야에 활용

**다항식 보간법**

$$
y = a_{2}x^2 + a_{1}x + a_{0}
$$

주어진 세 점 $(x, y)$에 대해 고유한 계수 $a_{0}, a_{1}, a_{2}$를 계산하여 이 점들을 가장 잘 맞추는 다항식을 찾는 문제로 함수 근사 문제로 볼 수 있습니다. 

#### **PCA & LDA**
고차원 데이터를 더 낮은 차원의 공간으로 변환하여 모델 학습이나 분석에서 중요한 정보를 유지하는 차원축소함

- PCA
    - 데이터를 가장 잘 설명할 수 있는 방향으로 차원을 축소
        - 데이터의 분산을 최대화하는 방향으로 축소하므로 원래 분포를 최대한 유지하면서 차원을 축소 
    - 비지도 학습 차원 축소 방법으로 많이 사용 
- LDA  
    - 차원 축소 후 분류 성능을 최적화하는 방향으로 데이터를 변환
        - 클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하는 방향으로 축소하여 분류가 잘 되도록 만듦
    - 클래스 정보를 활용하므로 지도 학습 차원 축소 기법을 사용

#### **Overfitting**
모델 성능 개선 및 과적합 방지 방법 
- 데이터 추가 
    - 학습 데이터 양 늘리기 
- 특성 수 줄이기 
    - 오토인코딩(Autoencoding)
        - 신경망을 이용해 중요 정보를 유지하면서 차원을 축소 
    - 드롭아웃(Dropout) 
        - 학습시 일부 뉴런을 무작위로 비활성화하여 모델이 더 견고한 특성을 학습하도록 함
- 정규화(Regularization)
    - 큰 가중치에 대한 패널치를 추가하여 모델이 데이터에 과도하게 맞추지 않도록 유도 

## **RNN**
시계열 데이터와 같은 순차적 데이터(Sequence Data) 처리를 위한 신경망으로 이전 시점의 정보와 현재 입력을 기반으로 다음 예측을 수행
- RNN은 이전 단어 또는 프레임을 기억하여 문맥을 이해하고 예측에 반영
- 단어를 문장 내 위치와 관계된 이전 및 다음 단어의 의미를 고려하여 문맥 이해 
- Applications
    - One-to-Many  
        - 한 개의 입력에서 여러 개의 출력을 생성 
        - Image Captioning
            - 이미지 하나에 여러 단어로 된 캡션 생성
    - Many-to-One
        - 여러 입력을 받아 하나의 출력으로 예측 
        - Sentiment Analysis
            - 텍스트 전체를 분석하여 긍정/부정 감정을 예측
    - Many-to-Many
        - 여러 입력을 받아 여러 개의 출력으로 예측
        - Machine Translation 
            - 문장 전체를 번역하여 다른 언어로 변환
        - Video Classification on Frame Level 
            - 비디오의 각 프레임을 개별적으로 분석하여 일련의 설명을 생성, 비디오 내용을 요약

## **강화학습**
### **Reinforcement Learning (RL)** 
최적의 행동 시퀀스를 찾는 문제에 적합
- Reinforcement Learning (RL)
    - 에이전트가 보상을 통해 최적의 행동 전략을 학습하는 방식 
- Deep Reinforcement Learning (DRL)
    - 복잡한 환경에서도 행동의 선택과 보상의 예측을 효과적으로 학습

| 항목 | RL (Reinforcement Learning) | DRL (Deep Reinforcement Learning) |
|-----|-----|-----|
| **장점**  | Optimal (최적의 정책 학습)<br> Fast Computation (빠른 계산)              | Optimal (복잡한 문제에서 최적화 가능)<br> Fast Computation (병렬처리 가능) |
| **단점**  | Pseudo-Polynomial (시간 복잡도가 입력 값에 따라 달라짐)<br> Non-Optimal (최적화가 어려울 수 있음) | Pseudo-Polynomial (높은 계산 비용)<br> Non-Optimal (훈련 불안정성) |

## **LLM**
대규모 데이터셋을 기반으로 학습하여 텍스트를 이해하고 생성할 수 있는 모델
### **LLM의 역사**
1. 언어 모델(LM: Language Model)
    - 통계적 언어 모델(SLM)
    - 인공신경망 기반 언어 모델
2. Transformer와 Attention 알고리즘
    - Transformer
        - 인코더-디코더 구조
    - Attention
        - 문맥 속에서 중요한 단어에 더 많은 가중치를 두어 긴 문장에서도 단어 간의 상관관계를 효율적으로 학습
3. 대규모 언어 모델(LLM: Large Language Model)
    - Bert
    - GPT
4. 초거대 AI
    - GPT-3
    - LaMDA

### **ChatGPT**
ChatGPT는 Generative Pre-trained Transformer 모델을 대화형으로 최적화한 것입니다. <br><br>
Generative: 텍스트를 생성하는 모델로 주어진 문맥에 맞는 다음 단어를 예측 <br>
Pre-trained: 대규모 텍스트 데이터를 사전 학습하여 기본적인 언어 패턴을 익힘 <br>
Transformer: 인코더-디코더 구조를 기반으로 하는 신경망 아키텍처로 Self-Attention 메커니즘을 활용하여 문맥 정보를 효과적으로 처리

- ChatGPT Training
    - 대화형 상호작용 방식을 통해 훈련
- 학습방법 
    - Reinforcement Learning by Human Feedback (RLHF)  
        - 인간의 피드백을 통해 모델의 출력을 개선하는 방식으로 학습 효율을 높임
    - Supervised Fine-Tuning (SFT)
        - 감독 학습을 통해 모델을 미세 조정하여 특정 작업에 맞게 최적화
    - Reward Model (RM)
        - 사용자 피드백에 따라 보상 모델을 통해 학습 성과를 평가하고 성능을 개선
    - Proximal Policy Optimization (PPO)
        - 이 방법을 사용하여 정책을 최적화
        - 강화학습에 사용하는 기법으로 최적의 의사결정을 내리기 위한 방법 