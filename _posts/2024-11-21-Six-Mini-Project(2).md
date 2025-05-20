--- 
title: "6차 미니 프로젝트 (2) | Six Mini Project (2)" 
date: 2024-11-25 17:58:45 +0900
achieved: 2024-11-21 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Deep Learning, Language Intelligence, Mini Project]
---
---------- 	
> KT 에이블스쿨 6차 미니프로젝트(2)를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
응급상황 자동 인식 및 응급실 연계 서비스

## **데이터셋**
- 자체 제작한 음성 데이터 및 text 데이터
- 중증도 분류 표 
- 전국 응급실 정보 

## **배경소개 및 비즈니스 상황**
1. 음성 인식 및 요약 
    - 응급전화를 받아 음성을 자동 인식 
    - 내용 요약/키워드 도출
2. 응급 상황 등급 분류 
    - 요약된 내용을 바탕으로 응급 상황 등급 분류 
3. 응급실 연계(추천)
    - 응급등급, 교통, 응급실 상황 고려 
    - 응급실 연계 

## **음성 인식 및 요약**

![음성 인식 및 요약](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img01.png?raw=true)

- 단계별 오디오 파일을 생성
    - 응급환자 중증도 분류기준에 따라 각 단계별로 오디오 파일을 생성

![응급실 필수 데이터 수집](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img02.png?raw=true)

- 응급실 필수 데이터 수집 
    - 병원이름, 주소, 전화번호, 응급실 유무, 경도, 위도를 필수 데이터로 판단

## **응급상황 등급 분류 및 모델 학습**

![초기 단계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img03.png?raw=true)

- 초기 단계
  - 등급당 60개의 텍스트를 담은 데이터로 학습 진행
  - 낮은 정확도(67%)
  - 샘플 텍스트에 분류 또한 제대로 이뤄지지 않음

![통합 및 수정](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img04.png?raw=true)

- 통합 및 수정
  - 라벨링이 부적절한 데이터를 수정하는 과정 진행
  - 개선된 정확도(74%)
  - 샘플 텍스트에 대한 분류에서의 아쉬움

![최종단계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img05.png?raw=true)

- 최종단계
  - 데이터의 불균형 해소
    - 3등급을 4등급으로 잘 못 분류하는 경우 환자의 생명에 위험을 초래할 수 있다고 판단하여 4단계로 분류된 경우 등급 별 확률을 분석하여 등급을 재확인
    - 4등급과 5등급의 분류가 제대로 이뤄지지 않아서 낮은 정확도(72%)
      - Confusion Matrix를 확인해보면 응급환자를 비응급환자로 판단하는 경우를 8회에서 4회로 감소 

## **지도를 이용하여 가장 가까운 응급실 연계**

![응급실 3곳 추천](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img06.png?raw=true)

- 응급환자에게 가장 가까운 응급실 연계 
  - Get haversine을 통해 위도와 경도를 기반으로 직선 거리를 계산하여 진행
  - 이후 haversine거리가 짧은 상위 10개 병원에 대해 naver map api를 통해 실시간 도로 환경을 반영하여 최소 거리 구함
  - 가까운 응급실 3곳 추천할 수 있도록 시스템 구축

## **모델 실행 및 결과**
### **파이프라인**
```python
# audio : 'audio.mp3'
# la, lo : float, float
def pipeline(audio, la, lo) :
  print("0 / 8...")
  # 기본 데이터 로딩
    # 0. Path 지정
  path = '/content/drive/MyDrive/project6_2/'

    # 1. GPT-3.5, API_KEY 불러오기
  with open(path + 'api_key.txt', 'r') as file:
    openai.api_key = file.readline().strip()

    # 2. Directions5 API_ID, API_KEY 불러오기
  with open(path + 'map_key.txt', 'r') as file:
    data = json.load(file)
  c_id = data['c_id']
  c_key = data['c_key']

    # 3. 응급실 데이터프레임 불러오기
  emergency_df = pd.read_csv(path + 'df_hos.csv')

    # 4. BERT Model 불러오기
  save_dir = path + 'fine_tuned_bert'

  print("1 / 8... \t|\t모델 불러오는 중")
  # 모델 로드
  model = AutoModelForSequenceClassification.from_pretrained(save_dir)
  tokenizer = AutoTokenizer.from_pretrained(save_dir)

  print("2 / 8... \t|\tOPEN AI KEY 등록 중")
  # open ai key 등록
  em.register_key(openai.api_key)

  print("3 / 8... \t|\t음성을 문장으로 변환 중")
  # Audio to Text
  text = em.audio_to_text(path+'audio/', audio)

  print("4 / 8... \t|\t문장 요약 중")
  # Text to Summary
  summary = em.text_summary(text)

  print("5 / 8... \t|\t등급 분류 중")
  # 응급실 등급 분류
  emer_lev, prob = em.predict(summary, model, tokenizer)

  print("6 / 8... \t|\t등급 확인 중")

  # (OPT) 등급에 따른 응급실 호출 여부 로직 구현
  if emer_lev == 4 :
    if prob[0][0] + prob[0][1] + prob[0][2] > prob[0][3] +prob[0][4] :
      print(f'KTAS 등급:{emer_lev}\t|\t응급상황입니다. 가까운 응급실 3곳을 선별합니다.')
    else :
      print(f'KTAS 등급:{emer_lev}\t|\t응급상황이 아닙니다.')
      return []
  else :
    if emer_lev < 4 :
      print(f'KTAS 등급:{emer_lev}\t|\t응급상황입니다. 가까운 응급실 3곳을 선별합니다.')
    else :
      print(f'KTAS 등급:{emer_lev}\t|\t응급상황이 아닙니다.')
      return []

  print("7 / 8... \t|\t응급실 선별 중")

  # 가까운 응급실 3곳 추천
  dist_list = em.emergency_recommendation(la, lo, emergency_df, c_id, c_key)

  print("8 / 8... \t|\t완료")
  return dist_list
``` 
### **결과**

![응급실 3곳 추천](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img07.png?raw=true)

![응급실 3곳 추천](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject6_2_img08.png?raw=true)

- 오디오 데이터와 좌표를 받아 응급 상황을 분류 진행
- 응급상황인 경우 가장 가장 가까운 병원 3곳 추천

## **고찰**
프로젝트를 진행하면서 모델 성능에 가장 큰 영향을 미치는 요소는 학습 파라미터보다는 데이터셋의 구성과 퀄리티라는 점을 깨달았습니다. 라임 기법을 사용해 문장에서 중요한 키워드에 가중치를 부여하고, 문장의 민감도와 특정 단어의 중요도를 평가하는 과정에서 데이터셋의 구성과 민감도가 성능에 미치는 영향을 더욱 실감할 수 있었습니다.<br>
저희 팀은 응급환자를 놓치지 않도록 모델 개발에 중점을 두었기 때문에 3등급에 대한 리콜을 높이는데 성공하였으나 각 등급별 다중 분류에서 정확도가 다소 떨어지는 문제가 있습니다. 이 부분을 개선하면 좋을 것 같다는 생각이 듭니다. 