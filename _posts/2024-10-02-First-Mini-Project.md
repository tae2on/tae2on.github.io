--- 
title: "1차 미니 프로젝트 | First Mini Project" 
date: 2024-10-02 19:44:45 +0900
achieved: 2024-09-25 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Analysis, Mini Project]
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
![정류장수와 승차총승객수의 관계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/bus_stop_passenger.jpg?raw=true)
<p align="center">정류장수와 승차총승객수의 관계</p>

- 강남구와 관악구와 같이 정류장수보다 승차총승객수가 높은 자치구는 대중교통 수요가 높고 유동인구가 많은 것으로 판단됩니다. 이러한 지역에는 추가적인 대중교통 설치를 고려할 필요가 있어보입니다. 
- 그에 반해 강동구와 강복구 같이 정류장수보다 승차총승객수가 작은 자치구는 대중교통의 이용이 저조한 걸로 보이며 다른 교통수단(자가용, 택시, 자전거 등)을 이용할 가능성이 높아보입니다. 

![자치구별 노선수와 승차평균승객수의 관계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/bus_stop_passenger_district.jpg?raw=true)
<p align="center">자치구별 노선수와 승차평균승객수의 관계</p>

- 모든 자치구에서 노선수보다 승차평균승객수가 높은 걸로 보아 대중교통의 이용이 높아보입니다. 
- 관악구, 동대문구, 용산구와 같이 노선수보다 승차평균승객수가 높은 지역은 노선수가 부족하여 보입니다. 서비스 확대가 필요해 보입니다. 

![자치구별 노선수와 승차 평균승객수의 관계](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/boarding_alighting_comparison_districts.jpg?raw=true)
<p align="center">자치구별 노선수와 승차 평균승객수의 관계</p>

- 대부분 자치구에서 승차총승객수와 하차총승객수가 비슷한 패턴을 보입니다. 

### **구별 유동인구 분석** 
![자치구별 평균 이동 시간과 이동 인구(합) 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/average_travel_time_population_comparison_districts.jpg?raw=true)
<p align="center">자치구별 평균 이동 시간과 이동 인구(합) 비교</p>

- 강남구, 서초구와 송파구 외에는 이동인구(합)보다는 이동평균(분)이 더 낮은 걸로 보입니다. 이는 해당 자치구에서 교통 서비스가 비교적 원활하게 운영되고 있음을 알 수 있습니다.  

### **구별 주민등록인구 분석** 
![자치구별 남자와 여자 인구수 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/district_male_female_population_trend.jpg?raw=true)
<p align="center">자치구별 남자와 여자 인구수 비교</p>

- 자치구별 남녀 비율이 거의 비슷한 걸로 보입니다. 
- 송파구에 사람이 제일 많고 종로구에 사람이 제일 적은 걸 볼 수 있습니다. 

### **구별 업종등록 분석**

![자치구별 업종 수 비교](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/district_business_count_trend.jpg?raw=true)
<p align="center">자치구별 업종 수 비교</p>

- 강남구에 모든 업종들이 높은 수치를 띄고 있는 것을 볼 수 있습니다. 
- 노원구, 양천구만 일반 교과 학원이 한식 일반 음식점업보다 높은 수치가 나타나는 것으로 보아 이 자치구에 학생들이 많이 포진되어 있을 것으로 예상됩니다. 

## **팀과제: 데이터 분석 및 인사이트 도출** 
개인과제의 데이터 분석을 바탕으로 팀원들과 가설을 세워 가설을 검정하고 결론 도출
### **가설 수립**
가설1: 인구 수가 많을수록 노선 수가 많을 것이다. <br>
가설2: 상권이 발달된 자치구의 노선 수가 많을 것이다.<br>
가설3: 평균 이동 시간이 길수록 노선 수가 적을 것이다.<br>
가설4: 승하차 인구가 많을수록 노선 수가 많을 것이다.

### **가설 검정**
#### **가설1: 인구 수가 많을수록 노선 수가 많을 것이다.**
![가설1: 단변량 분석 & 이변량 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject1_team_img1.jpg?raw=true)
<p align="center">가설1: 단변량 분석 & 이변량 분석</p>

- 인구수가 많을수록 이동하는 교통량은 많아질 수 있습니다. 따라서 인구수와 노선수는 비례한다는 사실을 알 수 있었습니다. 
- 상관계수, p-value와 산점도 모두 확인하였을 때 약한 상관계수(0.313)로, 인구수와 노선수는 아무런 관계가 없다는 결과를 볼 수 있었습니다.

#### **가설2: 상권이 발달된 자치구의 노선 수가 많을 것이다.**
![가설2: 단변량 분석 & 이변량 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject1_team_img2.jpg?raw=true)
<p align="center">가설2: 단변량 분석 & 이변량 분석</p>

경험적 근거와 사고실험을 통하여 여러 업종 중에서 한식 일반 음식점업, 커피전문점, 일반 교과 학원, 기타주점업이 노선수와 유의미한 관계가 있을거라고 가설을 설정하였습니다. 

**한식 일반 음식접엄**
- 상관계수와 p-value 모두 확인하였을 때 상관계수(0.47)는 95% 신뢰 구간 내에서 유의미한 양의 상관관계를 보였습니다.

**커피전문점**
- 상관계수와 p-value를 확인하였을 때 상관계수(0.48)는 중간 정도의 상관관계를 보였습니다.

**일반교과 학원 & 기타주점업**
- 상관계수와 p-value를 확인하였을 때 0.29, 0.1 수치의 상관계수를 가지며 유의미한 관계를 찾기 어려웠습니다. 
- 신뢰구간도 넓게 형성되어 있어 통계적으로 유의미하지 않음을 알 수 있었습니다. 

#### **가설3: 평균 이동 시간이 길수록 노선 수가 적을 것이다.**
![가설3: 단변량 분석 & 이변량 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject1_team_img3.jpg?raw=true)
<p align="center">가설3: 단변량 분석 & 이변량 분석</p>

- 산점도를 분석하였을 때 뚜렷하지 않지만 평균이동시간이 낮을수록 노선수가 높은 모습을 확인할 수 있었습니다. 
- 상관계수와 p-value를 확인하였을 때 상관계수(-0.52)는 중간 정도의 상관관계를 보였습니다. 

#### **가설4: 승하차 인구가 많을수록 노선 수가 많을 것이다.**
![가설4: 단변량 분석 & 이변량 분석](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject1_team_img4.jpg?raw=true)
<p align="center">가설4: 단변량 분석 & 이변량 분석</p>

- 산점도 분석을 통해 승하차 총승객수가 많을수록 노선수가 증가하는 경향을 확인할 수 있었습니다. 
- 상관계수와 p-value를 검토한 결과, 상관계수는 강한 양의 상관관계(0.72)를 보였습니다.

### **결론** 
선정한 가설 중 상관계수가 0.5 이상인 관계를 가지는 데이터를 기준으로 산점도를 분석하였습니다.

- 가설에서 공통적으로 노선수가 적다고 판단되는 자치구를 우선적으로 노선을 만들어야 한다고 판단하였습니다. <br>
1순위: 강동구, 중랑구<br>
2 순위: 광진구<br>
3 순위: 송파구, 강서구, 양천구, 성동구<br>
4 순위: 관악구, 용산구, 금천구