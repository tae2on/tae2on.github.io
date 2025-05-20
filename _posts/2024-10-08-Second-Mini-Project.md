--- 
title: "2차 미니 프로젝트 | Second Mini Project" 
date: 2024-10-08 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Processing, Machine Learning, Mini Project]
---
---------- 	
> KT 에이블스쿨 2차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
딥러닝 모델링을 통한 신규임대아파트 단지의 등록 차량수 예측

## **데이터셋**
- train.xlsx, test.xlsx
- 단지별 데이터와 단지 상세 데이터가 포함

## **개인과제**
도메인 이해 및 데이터 전처리, 데이터 분석
### **데이터 전처리** 
- 결측치 채우기 (최빈값)
- 변수 추가 (준공연도, 총면적)
- 불필요한 변수 제거 (단지명, 단지내주차면수, 준공일자)
- 데이터 분리 (단지별 데이터 분리, 상세 데이터 분리)
- 상세 데이터 집계 (단지코드별 총면적 합, 전용면적 구간별(피벗 형태))
- 임대보증금, 임대료 평균 집계
- 집계 결과 합치기 
- 변수 값 범주값으로 변경 (난방방식, 승강기설치여부)
- 불필요한 변수 제거 (단지코드, 지역)

![데이터 전처리](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_01.jpg?raw=true)
<p align="center">원본 데이터와 데이터 전처리 이후의 비교</p>

### **상관분석**

![상관분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_04.jpg?raw=true)
<p align="center">상관분석</p>

숫자형 변수들과 실차량수의 상관분석을 통해 강한 관계를 갖는 변수 선정 
- 총세대수(0.71)
- 총면적(0.82)

### **단변량 분석**
#### **총세대수 수치화**
![총세대수 수치화](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_02.jpg?raw=true)
<p align="center">총세대수 수치화</p>

#### **총면적 수치화**
![총면적 수치화](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_03.jpg?raw=true)
<p align="center">총면적 수치화</p>

### **이변량 분석**
#### **총세대수와 실차량수의 관계**
![총세대수와 실차량수의 관계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_06.jpg?raw=true)
<p align="center">총세대수와 실차량수의 관계</p>

- 총세대수와 실차량수는 양의 상관관계를 갖는 것을 볼 수 있습니다. 
- 거의 회귀선 근처에 값들이 분포하는 것을 보아 강한 상관관계로 볼 수 있습니다. 

#### **총면적과 실차량수의 관계**
![총면적과 실차량수의 관계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_05.jpg?raw=true)
<p align="center">총면적과 실차량수의 관계</p>

- 총면적과 실차량수는 양의 상관관계를 갖는 것을 볼 수 있습니다. 
- 거의 회귀선 근처에 값들이 분포하는 것을 보아 강한 상관관계로 볼 수 있습니다. 

### **범주형 변수와 실차량수의 관계 분석**
#### **지역별 평균 실차량수**
![지역별 평균 실차량수](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_07.jpg?raw=true)
<p align="center">지역별 평균 실차량수</p>

- 세종이 실차량수가 가장 높은 걸로 보아 세종에는 주차 문제가 많을 것으로 예상됩니다. 
- 충북, 전북, 제주 순으로 가장 낮은 걸로 보아 상대적으로 주차 문제가 적을 것으로 보입니다. 

#### **준공연도별 평균 실차량수**
![준공연도별 평균 실차량수](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_08.jpg?raw=true)
<p align="center">준공연도별 평균 실차량수</p>

- 2017년도부터 점점 수치가 떨어지는 걸로 보아 최근 아파트 단지들이 준공될 때, 주차 공간 확보가 충분하지 않을 수 있다는 것으로 예상됩니다. 

#### **건물형태별 평균 실차량수**
![건물형태별 평균 실차량수](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_09.jpg?raw=true)
<p align="center">건물형태별 평균 실차량수</p>

- 복도식 건물이 평균 실차량수가 가장 낮은 걸로 보아 주차 공간이 상대적으로 여유 있을 가능성이 높아보입니다. 
- 계단식과 혼합식이 거의 비슷한 비율로 높은 수치를 가지고 해당 유형의 건물은 주차 공간이 부족해 보입니다. 

#### **난방방식별 평균 실차량수**
![난방방식별 평균 실차량수](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_10.jpg?raw=true)
<p align="center">난방방식별 평균 실차량수</p>

- 지역난방과 지역가스난방이 다른 난방 방식에 비해 압도적으로 높은 평균 실차량수를 보이고 있습니다.
- 이 두 가지 난방 방식이 신규 아파트 단지에서 널리 사용되고 있음으로 이러한 단지들은 인구 밀도가 높아 주차 공간 부족 문제가 심화될 가능성이 있습니다.

#### **승강기설치여부별 평균 실차량수**
![승강기설치여부별 평균 실차량수](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_11.jpg?raw=true)
<p align="center">승강기설치여부별 평균 실차량수</p>

- 승강기가 없는 단지에서는 평균 실차량수가 낮은 경향을 띄고 있는 것을 보아 상대적으로 세대 수가 적거나 세대당 차량 보유수가 작다는 것으로 예상됩니다. 

## **팀과제** 
개인과제의 데이터 분석을 바탕으로 팀원들과 모델링을 통한 예측
### **가변수화**
![가변수화](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img1.jpg?raw=true)
<p align="center">가변수화</p>
 
건물형태, 난방방식 변수에 대해 가변수화 수행
### **머신러닝 모델링**
#### **Decision Tree**
![Decision Tree](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img2.jpg?raw=true)
<p align="center">Decision Tree: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'max_depth': 3}
- 예측성능: 0.6651537158010903
- MAE: 174.65043205396978
- R²: 0.6113863442949945

#### **KNN**
![KNN](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img3.jpg?raw=true)
<p align="center">KNN: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'n_neighbors': 4}
- 예측성능: 0.6761371915901122
- MAE: 164.60096153846155
- R²: 0.6129804776167364

#### **Random Forest**
![Random Forest](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img4.jpg?raw=true)
<p align="center">Random Forest: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'max_depth': 4}
- 예측성능: 0.7574485715801873
- MAE: 138.03102758238398
- R²: 0.6876475342158248

#### **XGBoost**
![XGBoost](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img5.jpg?raw=true)
<p align="center">XGBoost: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'max_depth': 1}
- 예측성능: 0.7235627955495929
- MAE: 133.9036045074463
- R²: 0.6917700925614235

#### **LightGBM**
![LightGBM](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img6.jpg?raw=true)
<p align="center">LightGBM: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'max_depth': 2}
- 예측성능: 0.7462778686736506
- MAE: 136.7796400536461
- R²: 0.6728714050335658

#### **GreedSearch**
![GreedSearch](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img7.jpg?raw=true)
<p align="center">GreedSearch: 실제값과 예측값 비교 시각화</p>

- 파라미터: {'max_depth': 6}
- 예측성능: 0.7594187206084257
- MAE: 128.42782726454348
- R²: 0.7037096294257363

### **성능 비교** 
![GreedSearch](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img8.jpg?raw=true)
<p align="center">성능 비교</p>

Random Forest로 선정
- 높은 성능지표 
- 타 모델에 비해 파라미터값에 따른 성능 지표가 균일

### **test데이터의 예상차량수 예측하기**
#### **파이프라인 구축**
```python
import pandas as pd

def data_pipeline(data):
    # 데이터 복사
    apt01 = data.copy()
    # 결측치가 있는 열 목록
    columns_with_na = ['건물형태', '난방방식', '승강기설치여부']
    
    # 각 열의 최빈값으로 결측치 채우기
    for column in columns_with_na:
        mode_value = apt01[column].mode()[0]  # 각 열의 최빈값 계산
        apt01[column].fillna(mode_value, inplace=True)  # 최빈값으로 결측치 채우기

    # '준공연도' 변수 추가 (준공일자에서 앞 4자리 추출)
    apt01['준공연도'] = apt01['준공일자'].astype(str).str[:4].astype(int)
    
    # '총면적' 변수 추가 (전용면적 + 공용면적) * 전용면적별세대수
    apt01['총면적'] = (apt01['전용면적'] + apt01['공용면적']) * apt01['전용면적별세대수']
    
    # 불필요한 변수 제거
    apt01.drop(columns=['단지명', '단지내주차면수', '준공일자'], inplace=True)

    # 단지별 데이터 분리
    data01 = apt01[['단지코드', '총세대수', '지역', '준공연도', '건물형태', '난방방식', '승강기설치여부']].drop_duplicates()
    data01.reset_index(drop=True, inplace=True)

    # 상세 데이터 분리
    data02 = apt01[['단지코드', '총면적', '전용면적별세대수', '전용면적', '공용면적', '임대보증금', '임대료']]

    # 단지코드별 총면적 합 집계
    df_area = data02.groupby('단지코드')['총면적'].sum().reset_index()

    # 전용면적 구간 정의
    bins = [0, 10, 30, 40, 50, 60, 70, 80, 200]
    labels = ['면적0_10', '면적10_30', '면적30_40', '면적40_50', '면적50_60', '면적60_70', '면적70_80', '면적80_200']

    # 전용면적 구간 추가
    data02['전용면적구간'] = pd.cut(data02['전용면적'], bins=bins, labels=labels, right=False)

    # 단지코드와 전용면적구간별 전용면적별세대수 합 집계
    temp = data02.groupby(['단지코드', '전용면적구간'], as_index=False)['전용면적별세대수'].sum()

    # 피벗 형태로 변환
    df_pivot = temp.pivot(index='단지코드', columns='전용면적구간', values='전용면적별세대수')
    df_pivot.columns.name = None
    df_pivot.reset_index(inplace=True)

    # 단지코드별 임대보증금, 임대료 평균 집계
    df_rent = data02.groupby('단지코드', as_index=False).agg({
        '임대보증금': 'mean',
        '임대료': 'mean'
    })

    # 데이터프레임 조인
    base_data = data01.merge(df_area, on='단지코드', how='left') \
                      .merge(df_pivot, on='단지코드', how='left') \
                      .merge(df_rent, on='단지코드', how='left')

    # '난방방식' 변수 값 변경
    base_data['난방방식'] = base_data['난방방식'].replace({
        '개별가스난방': '개별',
        '개별유류난방': '개별',
        '지역난방': '지역',
        '지역가스난방': '지역',
        '지역유류난방': '지역',
        '중앙가스난방': '중앙',
        '중앙난방': '중앙',
        '중앙유류난방': '중앙'
    })

    # '승강기설치여부' 변수 값 변경
    base_data['승강기설치여부'] = base_data['승강기설치여부'].replace({
        '전체동 설치': 1,
        '일부동 설치': 0,
        '미설치': 0
    })

    # '단지코드'와 '지역' 변수 제거
    base_data.drop(columns=['단지코드', '지역'], inplace=True)

    # 가변수화 대상: '건물형태', '난방방식'
    dumm_cols = ['건물형태', '난방방식']

    # 가변수화 수행
    base_data = pd.get_dummies(base_data, columns=dumm_cols, drop_first=True, dtype=int)

    return base_data
```

![파이프라인을 통한 전처리](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img10.jpg?raw=true)
<p align="center">파이프라인을 통한 전처리</p>

#### **예측 결과 확인**
test 데이터셋을 파이프라인을 통해 전처리한 후 성능이 가장 좋은 모델(Random Forest)로 예측
![예측 결과 확인](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img11.jpg?raw=true)
<p align="center">예측 결과 확인</p>

#### **아파트 기본 정보에 예상차량수 추가**

![아파트 기본 정보에 예상차량수 추가](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject2_team_img9.jpg?raw=true)
<p align="center">아파트 기본 정보에 예상차량수 추가</p>

## **고찰**
팀원들과 모델링을 하기 전에 데이터의 수치를 가지고 여러 의견을 공유하며 더 넓은 시각을 가질 수 있었습니다.<br>
모델링 과정에서 다양한 알고리즘의 성능을 비교하며 이론적으로 알고 있던 내용을 실전에 적용할 수 있는 좋은 기회가 되었습니다.<br>
팀원들과 모델링 과정을 서로 공유하며 직접 해보지 않았던 모델링의 결과에 대해서도 알 수 있어 더욱 좋은 성능을 가진 모델을 선정할 수 있었습니다.<br>
이러한 협업을 통해 다양한 관점을 고려하고 여러 모델의 장단점을 비교 분석하여 최적의 결과를 도출할 수 있었습니다. 