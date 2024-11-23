--- 
title: "6차 미니 프로젝트 (1) | Six Mini Project (1)" 
date: 2024-11-23 13:39:45 +0900
achieved: 2024-11-12 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Language Intelligence, Mini Project]
---
---------- 	
> KT 에이블스쿨 6차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
시계열데이터 모델링을 통한 상품별 판매량 예측

## **데이터셋**
- oil_price: 일별 유가 데이터
- orders: 일별 매장별 고객 방문 수 
- sales: 판매 정보
- products: 상품 기본 정보 
- stores: 매장 기본 정보

## **배경소개 및 비즈니스 상황**
- 유통 매장에서 상품별 재고문제를 AI 기반 수요량 예측 시스템 개발을 통해 해결 
    - 44번 매장의 핵심 상품 3개를 선정한 후 선정 후, 수요 예측을 기반한 발주 시스템의 가능성을 검토 
- 고객사의 주요 매장(ID-44)의 핵심 상품에 대한 수요량을 예측하고 재고를 최적화 
    - 매일 저녁 9시 당일 판매가 마감된 후, 상품별 리드타임에 맞게 판매량 예측

## **데이터 탐색**
### **전체 데이터 확인**

**대상 매장(44)의 방문자 추이**
![방문자 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img01.png?raw=true)

**상품(3, 12, 42) 판매량 추이**
![상품별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img02.png?raw=true)

**요일별 전체 판매량 비교**
![요일별 판매량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img03.png?raw=true)

**월별 전체 판매량 비교**

![월별 전체 판매량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img12.png?raw=true)

#### **인사이트 도출**
- 전체적으로 12번 상품이 제일 판매되고 42번 상품이 가장 적게 판매된 걸 확인할 수 있다. 
- 일요일이 평균적으로 판매량이 제일 많다.
    - 판매량 순서: 일요일 > 토요일 > 월요일 > 금요일 > 화요일 > 수요일 > 목요일 
- 월별 판매량은 12월, 11월, 10월, 9월 순으로 연말에 높고 6월, 5월, 3월, 4월 순으로 낮은 걸 보인다.
    - 미국은 집에서 파티하는 문화가 있고 블랙프라이데이와 같은 연말 할인행사가 크게 작용하는 것으로 보임
    - 연말에 카드실적을 채우고자 방문한다고 생각
- 방문자가 늘면 판매량도 늘어났다. 
    - 미국은 택배 문화가 엄청 발달하지도, 2014 ~ 2017년이면 더 그럴 것으로 판단

### **3번 상품 패턴 확인** 

**년도별 판매량 추이**
![년도별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img04.png?raw=true)

**동일 카테고리의 상품별 판매량 추이**
![동일 카테고리의 상품](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img15.png?raw=true)
![동일 카테고리의 상품별 판매량](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img05.png?raw=true)

**휘발유 가격과 상품 판매량 추이 비교**
![휘발유 가격과 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img06.png?raw=true)

**방문 고객수와 상품 판매량 추이 비교**
![방문 고객수와 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img07.png?raw=true)


**다른 매장과의 판매량 비교**
![다른 매장과의 판매량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img08.png?raw=true)

**계절별 판매량 추이**

![계절별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img09.png?raw=true)


**월별 판매량 분석**

![월별 판매량 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img10.png?raw=true)

**시계열 데이터 분해**
![시계열 데이터 분해](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img11.png?raw=true)

#### **3번 상품 인사이트 도출**
3번 상품: 음료 
- 토요일과 일요일마다 높은 판매량을 보인다.
- 월별 판매량은 1월, 7월, 9월, 10월, 11월, 12월에 높고 2월, 3월, 4월, 5월, 6월, 8월에 낮다. 
    - 가을, 겨울에 판매량이 증가하는 경향이 보임
        - 이 계절에 할로윈과 크리스마스와 같은 행사/연휴가 많아 영향을 준다고 판단
    - 16년 이후로는 이런 경향이 약해짐
- 음료는 맥주와 같은 기호품으로 추청된다.
- 2016년 4월 19일에 가장 높은 판매량을 기록하였다. 
- 유사하게 높은 판매량을 가지는 같은 카테고리 12번 상품은 여름과 봄에 강세를 보인다.

### **12번 상품 패턴 확인**
**년도별 판매량 추이**
![년도별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img13.png?raw=true)

**동일 카테고리의 상품별 판매량 추이**
![동일 카테고리의 상품별 판매량](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img16.png?raw=true)

![동일 카테고리의 상품별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img14.png?raw=true)

**휘발유 가격과 상품 판매량 추이 비교**
![휘발유 가격과 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img17.png?raw=true)

**방문 고객수와 상품 판매량 추이 비교**
![방문 고객수와 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img18.png?raw=true)

**요일별 판매량 비교**

![요일별 판매량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img19.png?raw=true)

**요일별 방문자 수 비교**

![요일별 방문자 수 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img20.png?raw=true)

**계절별 판매량 비교**

![계절별 판매량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img21.png?raw=true)

**시계열 데이터 분해**
![시계열 데이터 분해](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img22.png?raw=true)

#### **12번 상품 인사이트 도출**
12번 상품: 우유
- 우유 판매량이 11월부터 오르기 시작해서 11월 말까지 증가하고 12월부터는 감소하는 경향이 있다. 
    - 우유의 경우 11월 ~ 1월 증감하는 경향을 보임 
    - 2016년 3월 ~ 2016년 5월은 잘 모르겠음
    - 우유의 판매량은 연말에 급감하였다가 다시 회복하는 추세를 보이지만 1월 1일 마트의 휴무일 데이터가 영향을 끼치는 듯 함 
    - 연말에 제품 주문량을 늘려야할 것으로 판단
- 동일한 카테고리를 분석해본 결과
    - 가공식품(prepared) 경우 딱히 경향을 가지고 있지 않음
    - 냉동식품(frozen)의 경우 다른 달보다 11월 ~ 1월에 증감하는 경향이 있음
    - 빵의 경우 일정한 패턴을 가지고 있진 않지만 증가하는 추세로 보여짐
    - 요거트의 경우 패턴을 찾기 힘듦

### **42번 상품 패턴 확인**

**년도별 판매량 추이**
![년도별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img23.png?raw=true)

**동일 카테고리별 판매량 추이**
![동일 카테고리별 판매량](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img25.png?raw=true)

![동일 카테고리별 판매량 추이](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img24.png?raw=true)

**휘발유 가격과 상품 판매량 추이 비교**
![휘발유 가격과 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img26.png?raw=true)

**방문 고객수와 상품 판매량 추이 비교**
![방문 고객수와 상품 판매량 추이 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img27.png?raw=true)

**원본과 MA 활용하여 비교**
![원본과 MA 활용하여 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img28.png?raw=true)

- MA: 일정 기간 동안의 평균값을 계산하여 데이터를 평활화 (이동평균)
- MA(이동평균)를 활용해서 그린 것이 추세를 보기에 더 좋아보인다.

**트렌드 분석**
![트렌드 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img29.png?raw=true)

**요일별 변화량 비교**
![요일별 변화량 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img30.png?raw=true)

**시계열 데이터 분해**
![시계열 데이터 분해](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img31.png?raw=true)

#### **42번 상품 인사이트 도출**
42번 상품: 농작물
- 매년 비슷한 패턴 가진다. 
    - 1월부터 7월까지 증가, 8월 말부터 9월로 갈수록 천천히 떨어지는 경향을 보임
- 1월 1일은 방문 고객 수가 0명으로 42번을 제외하고 Qty가 전부 0이다.
    - Minnesota를 확인해본 결과, 농업이 발달한 도시
        - 많은 주민들이 농업에 종사하며 긴 겨울과 짧은 여름의 계절성을 뜀
        - (가정) k-mart랑 협업해서 지역의 농업을 지원해주는 목적으로 공휴업에도 농업판매량이 기록된다고 가정할 수 있음
- 같은 카테고리끼리의 관계를 찾기가 힘들어보인다.
- 유가와 농산물 사이의 관계가 있다고 보기 힘들다.
- 방문 고객수와 식료품 판매량은 큰 관계가 없는 것으로 보인다. 
- 가격의 변화가 월요일부터 점점 커지는 것을 확인할 수 있다. 
    - 월요일에 가격이 많이 떨어지는 것을 확인 가능
    - 월요일에만 하락이 있고 다른 요일에는 하락이 없는 거 확인 가능 

## **데이터 전처리**
도출된 인사이트를 토대로 필요한 변수 추가 및 불필요한 변수 삭제

![데이터 전처리](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img33.png?raw=true)


```python
 def data_compose(id):
    df = sales.loc[(sales['Store_ID'] == 44) & (sales['Product_ID'] == id)]
    df = df.merge(orders, how='left', on=['Date', 'Store_ID'])
    df = df.drop(['Store_ID', 'Product_ID'], axis = 1)

    df['CustomerCount'] = df['CustomerCount'].fillna(0)

    df['Qty_RM7'] = df['Qty'].rolling(7, min_periods=1).mean()

    df['Qty_RM14'] = df['Qty'].rolling(14, min_periods=1).mean()

    df['Customer_RM7'] = df['CustomerCount'].rolling(7, min_periods=1).mean()

    df['Customer_RM14'] = df['CustomerCount'].rolling(14, min_periods=1).mean()

    # 이틀 후(예측일)의 1주일 전 Qty와 CC
    df['Qty_before_5d'] = df['Qty'].shift(5)
    df['Qty_before_5d'] = df['Qty_before_5d'].fillna(df['Qty'])

    df['Customer_before_5d'] = df['CustomerCount'].shift(5)
    df['Customer_before_5d'] = df['Customer_before_5d'].fillna(df['CustomerCount'])

    df['Qty_before_7d'] = df['Qty'].shift(7)
    df['Qty_before_7d'] = df['Qty_before_7d'].fillna(df['Qty'])

    df['Customer_before_7d'] = df['CustomerCount'].shift(7)
    df['Customer_before_7d'] = df['Customer_before_7d'].fillna(df['CustomerCount'])

    df['Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x in [5, 6] else 0)
    df['Season'] = df['Date'].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})

    df = pd.get_dummies(df, columns=['Season'], drop_first=True)
    df = df.replace({False: 0, True: 1})

    df['Holiday'] = df['Date'].apply(lambda x: 1 if x.month == 1 and x.day == 1 else 0)

    df['y'] = df['Qty'].shift(-2)
    # df['y'] = df['y'].fillna(method='bfill')

    df = df.iloc[:-2]

    df = df.set_index('Date')

    return df
```

## **Baseline 모델**
### **LSTM 구조** 
![Baseline 모델_LSTM](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img32.png?raw=true)

- 3번 상품 / 12번 상품 / 42번 상품에 대한 평가지표
    - RMSE: 0.098 / 0.056 / 0.106
    - MAE : 0.071 / 0.046 / 0.084
    - $R^2$ : 0.53 / 0.55 / 0.34
    - MAPE : 0.15 / 0.16 / -0.095

### **CNN 구조** 
![Baseline 모델_CNN](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img34.png?raw=true)

- 3번 상품 / 12번 상품 / 42번 상품에 대한 평가지표
    - RMSE: 0.12 / 0.063 / 0.135
    - MAE : 0.078 / 0.053 / 0.102
    - $R^2$ : 0.344 / 0.433 / -0.754
    - MAPE : 0.15 / 0.2 / 0.371

## **LSTM 모델 튜닝**
![LSTM 모델 튜닝](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img35.png?raw=true)

- 3번 상품 / 12번 상품 / 42번 상품에 대한 평가지표
    - RMSE: 1904 / 2183 / 10.8
    - MAE : 1455 / 1735 / 8.53
    - $R^2$ : 0.73 / 0.51 / 0.49
    - MAPE : 0.13 / 0.15 / 0.09

## **최종결과**
![최종결과](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img36.png?raw=true)

![최종결과](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_1_img37.png?raw=true)

- 3번 상품 / 12번 상품 / 42번 상품
    - 일평균 재고량 : 7827 / 7325 / 68
    - 일평균 재고 금액 : 7827381 / 7325810 / 68819
    - 일평균 재고회전율: 1.392 / 1.423 / 1.604
    - 기회손실 수량 : -45912 / -35149 / -132
    
튜닝한 모델의 예측 그래프를 살펴보니 어느정도 성능이 보장된 것 같지만 실제 데이터와 확인해보니 성능이 잘 나오지 않았습니다. 

## **고찰**
팀원들과 전처리 방식을 거의 비슷하게 맞춰서 하였는데 서로 다른 전처리 방식을 통해 데이터셋을 구축하였다면 상품들의 특징과 연관성을 보다 다양하게 도출할 수 있었을 것 같아서 아쉬움이 남습니다. <br>
처음 미니프로젝트에 비해 가면갈수록 더 어렵다는 느낌이 확 다가왔던 미니 프로젝트였던 것 같습니다. 막히는 부분을 팀원들이 잘 알려줘서 이번 미니 프로젝트도 잘 끝낼 수 있었던 것 같습니다. 
