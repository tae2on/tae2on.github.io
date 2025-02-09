--- 
title: "Aice Associate 기출문제 풀이" 
date: 2025-01-23 20:46:45 +0900
achieved: 2025-01-23 19:48:27 +0900
math: true
categories: [Certification, Aice Associate]
tags: [Certification, KT Aivle School, python, Jupyter Notebook, Aice Associate]
---
---------- 	
> Aice Associate 기출 문제 풀이 정리본입니다. 
{: .prompt-info } 

## **샘플 문항 풀러가기**

🔗 [샘플 문항 풀러가기](https://aice.study/certi/practice)  

## **기출문제 풀이**
1. scikit-learn 패키지는 머신러닝 교육을 위한 최고의 파이썬 패키지입니다. scikit-learn를 별칭(alias) sk로 임포트하는 코드를 작성하고 실행하세요.

    ```python
    import sklearn as sk
    ```

2. Pandas는 데이터 분석을 위해 널리 사용되는 파이썬 라이브러리입니다. Pandas를 사용할 수 있도록 별칭(alias)을 pd로 해서 불러오세요.

    ```python
    import pandas as pd
    # import numpy as np
    # import seaborn as sns
    ```

3. 모델링을 위해 분석 및 처리할 데이터 파일을 읽어오려고 합니다. Pandas함수로 2개 데이터 파일을 읽고 합쳐서 1개의 데이터프레임 변수명 df에 할당하는 코드를 작성하세요.
    - A0007IT.json 파일을 읽어 데이터 프레임 변수명 df_a에 할당하세요.
    - signal.csv 파일을 읽어 데이터 프레임 변수명 df_b에 할당하세요.
    - df_a와 df_b 데이터프레임을 판다스의 merge 함수를 활용하여 합쳐 데이터프레임 변수명 df에 저장하세요 
        - 합칠 때 사용하는 키(on): 'RID'
        - 합치는 방법(how): 'inner'

    ```python
    df_a = pd.read_json('./A0007IT.json')
    df_b = pd.read_csv('./signal.csv')
    df = pd.merge(df_a, df_b, on='RID', how='inner')
    ```

4. Address1(주소1)에 대한 분포도를 알아 보려고 합니다. Address1(주소1)에 대해 countplot그래프로 만드는 코드와 답안을 작성하세요.
    - Seaborn을 활용하세요.
    - 첫번째, Address1(주소1)에 대해서 분포를 보여주는 countplot 그래프 그리세요.
    - 출력된 그래프를 보고 해석한 것으로 옳지 않은 선택지를 아래에서 골라 '답안04' 변수에 저장하세요. (예: 답안04 = 4)
        - 1. Countplot그래프에서 Address1(주소1) 분포를 확인시 '경기도' 분포가 제일 크다. 
        - 2. Address1(주소1) 분포를 '인천광역시'보다 '서울특별시'가 더 크다.
        - 3. 지역명이 없는 '-'에 해당되는 row(행)가 2개 있다. 

    ```python
    import seaborn as sns

    sns.countplot(data=df, x='Address1')
    plt.show()

    df = df[df['Address1'] != '-']
    답안04 = 3
    ```    

5. 실주행시간과 평균시속의 분포를 같이 확인하려고 합니다. Time_Driving(실주행시간)과 Speed_Per_Hour(평균시속)을 jointplot그래프로 만드세요.
    - Seaborn을 활용하세요.
    - X축에는 Time_Driving(실주행시간)을 표시하고 Y축에는 Speed_Per_Hour(평균시속)을 표시하세요.

    ```python
    sns.jointplot(data=df, x='Time_Driving', y='Speed_Per_Hour')
    plt.show()
    ```

6. 위의 jointplot 그래프에서 시속 300이 넘는 이상치를 발견할 수 있습니다. 가이드에 따라서 전처리를 수행하고 저장하세요.
    - 대상 데이터프레임: df
    - jointplot 그래프를 보고 시속 300 이상되는 이상치를 찾아 해당 행(Row)을 삭제하세요.
    - 불필요한 'RID' 컬럼을 삭제하세요.
    - 전처리 반영 후에 새로운 데이터프레임 변수명 df_temp에 저장하세요.

    ```python
    df_temp = df[df['Speed_Per_Hour'] < 300]
    
    df_temp = df_temp.drop(['RID'], axis=1)
    df_temp.info()
    ```

7. 모델링 성능을 제대로 얻기 위해서는 결측치 처리는 필수입니다. 아래 가이드를 따라 결측치 처리하세요. 
    - 대상 데이터프레임: df_temp
    - 결측치를 확인하는 코드를 작성하세요.
    - 결측치가 있는 행(raw)를 삭제하세요.
    - 전처리 반영된 결과를 새로운 데이터프레임 변수명 df_na에 저장하세요.
    - 결측치 개수를 '답안07' 변수에 저장하세요. (예: 답안07 = 5)

    ```python
    df_temp.isnull().sum()
    df_na = df_temp.dropna(axis=0)
    답안07 = 2
    ```

8. 모델링 성능을 제대로 얻기 위해서 불필요한 변수는 삭제해야 합니다. 아래 가이드를 따라 불필요 데이터를 삭제 처리하세요.
    - 대상 데이터프레임: df_na
    - 'Time_Departure', 'Time_Arrival' 2개 컬럼을 삭제하세요.
    - 전처리 반영된 결과를 새로운 데이터프레임 변수명 df_del에 저장하세요.

    ```python
    df_del = df_na.drop(['Time_Departure', 'Time_Arrival'], axis=1)
    ```

9. 원-핫 인코딩(One-hot encoding)은 범주형 변수를 1과 0의 이진형 벡터로 변환하기 위하여 사용하는 방법입니다. 원-핫 인코딩으로 아래 조건에 해당하는 컬럼 데이터를 변환하세요.
    - 대상 데이터프레임: df_del
    - 원-핫 인코딩 대상: object 타입의 전체 컬럼
    - 활용 함수: Pandas의 get_dummies
    - 해당 전처리가 반영된 결과를 데이터프레임 변수 df_preset에 저장해 주세요.

    ```python
    cols = df_del.select_dtypes('object').columns
    df_preset = pd.get_dummies(data=df_del, columns=cols)
    ```

10. 훈련과 검증 각각에 사용할 데이터셋을 분리하려고 합니다. Time_Driving(실주행시간) 컬럼을 label값 y로, 나머지 컬럼을 feature값 y로, 나머지 컬럼을 feature값 X로 할당한 후 훈련데이터셋과 검증데이터셋으로 분리하세요. 추가로 가이드 따라서 훈련데이터셋과 검증데이터셋에 스케일링을 수행하세요.
    - 대상 데이터프레임: df_preset
    - 훈련과 검증 데이터셋 분리
        - 훈련 데이터셋 label: y_train, 훈련 데이터셋 Feature: X_train
        - 검증 데이터셋 label: y_valid, 검증 데이터셋 Feature: X_valid
        - 훈련 데이터셋과 검증 데이터셋 비율은 80:20
        - random_state: 42
        - Scikit-learn의 train_test_split 함수를 활용하세요.
    - RobustScaler 스케일링 수행
        - sklearn.preprocessing의 RobustScaler 함수 사용
        - 훈련데이터셋의 Feature는 RobustScaler의 fit_transform 함수를 활용하여 X_train 변수로 할당
        - 검증데이터셋의 Feature는 RobustScaler의 transform 함수를 활용하여 X_test 변수로 할당

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler

    x = df_preset.drop('Time_Driving', axis=1)
    y = df_preset['Time_Driving']
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

    robustScaler = RobustScaler()

    X_train_robust = robustScaler.fit_transform(X_train)
    X_test_robust = robustScaler.fit_transform(X_valid)
    
    X_train = pd.DataFrame(X_train_robust, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_robust, index=X_valid.index, columns=X_valid.columns)
    ```

11. Time_Driving(실주행시간)을 예측하는 머신러닝 모델을 만들려고 합니다. 의사결정나무(decision tree)와 랜덤포레스트(RandomForest)는 여러 가지 규칙을 순차적으로 적용하면서 독립 변수 공간을 분할하는 모형으로 분류(classification)와 회귀 분석(regression)에 모두 사용될 수 있습니다. 아래 가이드에 따라 의사결정나무(decision tree)와 랜덤포레스트(RandomForest) 모델 만들고 학습을 진행하세요.
    - 의사결정나무(decision tree)
        - 트리의 최대 깊이: 5로 설정
        - 노드를 분할하기 위한 최소한의 샘플 데이터수(min_samples_split): 3로 설정
        - random_state: 120로 설정
        - 의사결정나무(decision tree) 모델을 df 변수에 저장해 주세요.
    - 랜덤포레스트(RandomForest)
        - 트리의 최대 깊이: 5로 설정
        - 노드를 분할하기 위한 최소한의 샘플 데이터수(min_samples_split): 3로 설정
        - random_state: 120로 설정
        - 랜덤포레스트(RandomForest) 모델을 rf 변수에 저장해주세요.
    - 위의 2개의 모델에 대해 fit을 활용해 모델을 학습해 주세요. 학습 시 훈련데이터 셋을 활용해 주세요.

    ```python
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=3, random_state=120)
    dt.fit(X_train, y_train)
    
    rf = RandomForestRegressor(max_depth=5, min_samples_split=3, random_state=120)
    rf.fit(X_train, y_train)
    ```

12. 위 의사결정나무(decision tree)와 랜덤포레스트(RandomForest) 모델의 성능을 평가하려고 합니다. 아래 가이드에 따라 예측 결과의 mae(Mean Absolute Error)를 구하고 평가하세요.
    - 성능 평가는 검증 데이터셋을 활용하세요.
    - 11번 문제에서 만든 의사결정나무(decision tree) 모델로 y값을 예측(predict)하여 y_pred_dt에 저장하세요.
    - 검증 정답(y_valid)과 예측값(y_pred_dt)의 mae(Mean Absolute Error)를 구하고 dt_mae 변수에 저장하세요.
    - 11번 문제에서 만든 랜덤포레스트(RandomForest) 모델로 y값을 예측(predict)하여 y_pred_rf에 저장하세요.
    - 검증 정답(y_valid)과 예측값(y_pred_rf)의 mae(Mean Absolute Error)를 구하고 rf_mae 변수에 저장하세요.
    - 2개의 모델에 대한 mae 성능평가 결과를 확인하여 성능 좋은 모델 이름을 '답안12' 변수에 저장하세요.
        - 예: 답안12 = 'decisiontree' 혹은 답안12 = 'randomforest'

    ```python
    from sklearn.metrics import mean_absolute_error

    y_pred_dt = dt.predict(X_valid)
    dt_mae = mean_absolute_error(y_valid, y_pred_dt)
    print(dt_mae)

    y_pred_rf = rf.predict(X_valid)
    rf_mae = mean_absolute_error(y_valid, y_pred_rf)
    print(rf_mae)

    답안12 = 'randomforest'
    print(답안12)
    ```

13. Time_Driving(실주행시간)을 예측하는 딥러닝 모델을 만들려고 합니다. 아래 가이드에 따라 모델링하고 학습을 진행하세요.
    - Tensorflow framework를 사용하여 딥러닝 모델을 만드세요.
    - 히든레이어(hidden layer) 2개 이상으로 모델을 구성하세요.
    - dropout 비율 0.2로 Dropout 레이어 1개를 추가해 주세요.
    - 손실함수는 MSE(Mean Squared Error)를 사용하세요.
    - 하이퍼파라미터 epochs: 30, batch_size: 16으로 설정해 주세요.
    - 각 에포크마다 loss와 metrics 평가하기 위한 데이터로 x_valid, y_valid 사용하세요.
    - 학습정보는 history 변수에 저장해 주세요.

    ```python
   model = Sequential()
   model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(64, activation='relu'))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse', metrics='mse')
   history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_valid, y_valid))
    ```

14. 위 딥러닝 모델의 성능을 평가하려고 합니다. Matplotlib 라이브러리 활용해서 학습 mse와 검증 mse를 그래프로 표시하세요.
    - 1개의 그래프에 학습 mse과 검증 mse 2가지를 모두 표시하세요.
    - 위 2가지 각각의 범례를 'mae', 'val_mse'로 표시하세요.
    - 그래프의 타이틀은 'Model MSE'로 표시하세요.
    - X축에는 'Epochs'라고 표시하고 Y축에는 'MSE'라고 표시하세요.

    ```python
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.legend(['mse', 'val_mse'])
    plt.title('Model MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mse')
    ```