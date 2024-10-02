--- 
title: "1차 미니 프로젝트 | First Mini Project" 
date: 2024-10-01 17:01:45 +0900
achieved: 2024-09-23 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Web Crawling]
---
---------- 	
> KT 에이블스쿨 1차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
서울시 생활 정보 기반 대중교통 수요 분석을 통한 버스 노선 추가가 필요한 서울 시 내 자치구 선정

## **데이터셋**
- 서울시 공공데이터 포털 (서울시 버스노선별 정류장 승하차 인원 정보)
- 서울시 공공데이터 포털 (서울시 버스 정류소 위치정보)
- 서울 열린데이터 광장 (서울시 구별 이동 2024년 8월 데이터)
- 서울시 주민 등록 데이터 
- 서울시 구별 등록 업종 상위 10개 데이터 

## **개인과제 - 데이터 분석 및 인사이트 도출**
도메인 이해 및 데이터 분석을 통한 인사이트 도출
### **구별 버스정류장 분석** 
<!-- bus_stop_passenger -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\bus_stop_passenger.jpg" alt="bus_stop_passenger" width="500" height="200" />_정류장수와 승차총승객 수의 관계_ 

- 강남구와 관악구와 같이 정류장수보다 승차총승객수가 높은 자치구는 대중교통 수요가 높고 유동인구가 많은 것으로 판단됩니다. 이러한 지역에는 추가적인 대중교통 설치를 고려할 필요가 있어보입니다. 
- 그에 반해 강동구와 강복구 같이 정류장수보다 승차총승객수가 작은 자치구는 대중교통의 이용이 저조한 걸로 보이며 다른 교통수단(자가용, 택시, 자전거 등)을 이용할 가능성이 높아보입니다. 
<!-- bus_stop_passenger_district -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\bus_stop_passenger_district.jpg" alt="bus_stop_passenger_district" width="500" height="200" />_자치구별 노선수와 승차평균승객수의 관계_ 

- 모든 자치구에서 노선수보다 승차평균승객수가 높은 걸로 보아 대중교통의 이용이 높아보입니다. 
- 관악구, 동대문구, 용산구와 같이 노선수보다 승차평균승객수가 높은 지역은 노선수가 부족하여 보입니다. 서비스 확대가 필요해 보입니다. 
<!-- boarding_alighting_comparison_districts -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\boarding_alighting_comparison_districts" alt="boarding_alighting_comparison_districts" width="500" height="200" />_자치구별 노선수와 승차 평균승객수의 관계_ 

- 대부분 자치구에서 승차총승객수와 하차총승객수가 비슷한 패턴을 보입니다. 

### **구별 유동인구 분석** 
<!-- average_travel_time_population_comparison_districts -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\average_travel_time_population_comparison_districts" alt="average_travel_time_population_comparison_districts" width="500" height="200" />_꺽은선 그래프: 자치구별 평균 이동 시간과 이동 인구(합) 비교_ 

<!-- district_travel_population_comparison -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\district_travel_population_comparison" alt="district_travel_population_comparison" width="500" height="200" />_막대 그래프: 자치구별 평균 이동 시간 및 이동 인구 비교_ 

- 강남구, 서초구와 송파구 외에는 이동인구(합)보다는 이동평균(분)이 더 낮은 걸로 보입니다. 이는 해당 자치구에서 교통 서비스가 비교적 원활하게 운영되고 있음을 알 수 있습니다.  

### **구별 주민등록인구 분석**
<!-- district_male_female_population_trend -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\district_male_female_population_trend" alt="district_male_female_population_trend" width="500" height="200" />_꺽은선 그래프: 자치구별 남자와 여자 인구수 비교_ 

<!-- bar_graph_district_male_female_population_trend -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\bar_graph_district_male_female_population_trend" alt="bar_graph_district_male_female_population_trend" width="500" height="200" />_막대 그래프: 자치구별 남자와 여자 인구수 비교_ 

- 자치구별 남녀 비율이 거의 비슷한 걸로 보입니다. 
- 송파구에 사람이 제일 많고 종로구에 사람이 제일 적은 걸 볼 수 있습니다. 

### **구별 업종등록 분석**
<!-- district_business_count_trend -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\district_business_count_trend" alt="district_business_count_trend" width="500" height="200" />_막대 그래프: 자치구별 남자와 여자 인구수 비교_ 
<!-- bar_graph_district_business_count_trend -->
<img src="C:\Users\User\Documents\GitHub\tae2on.github.io\assets\img\bar_graph_district_business_count_trend" alt="bar_graph_district_business_count_trend" width="500" height="200" />_막대 그래프: 자치구별 남자와 여자 인구수 비교_ 

- 강남구에 모든 업종들이 높은 수치를 띄고 있는 것을 볼 수 있습니다. 
- 노원구, 양천구만 일반 교과 학원이 한식 일반 음식점업보다 높은 수치가 나타나는 것으로 보아 이 자치구에 학생들이 많이 포진되어 있을 것으로 예상됩니다. 

## **팀과제 - 데이터 분석 및 인사이트 도출** 
개인과제의 데이터 분석을 바탕으로 팀원들과 가설을 세워 가설을 검정하고 결론 도출
### 가설 수립 
#### 가설 
#### 가설