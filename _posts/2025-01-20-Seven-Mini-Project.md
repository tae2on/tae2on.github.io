--- 
title: "7차 미니 프로젝트 | Seven Mini Project" 
date: 2025-01-20 11:10:45 +0900
achieved: 2024-12-24 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Java, IntelliJ IDEA, Mini Project, Front-end, Back-end]
---
---------- 	
> KT 에이블스쿨 7차 미니프로젝트를 수행한 내용 정리한 글입니다. 
{: .prompt-info } 

## **개요**
응급상황 인식 및 응급실 연계 서비스 포탈

## **배경소개 및 비즈니스 상황**
1. 사용자 입력 
    - 사용자는 브라우저를 통해 시스템에 음성 메시지와 위치 정보를 입력
        - 음성을 통해 응급 상황을 설명 
        - 위치 정보는 응급실 추천을 위한 필수 데이터 
2. 프론트엔드 
    - Spring Boot 사용
    - Docker를 사용하여 컨테이너화
3. 백엔드
    - Spring Boot와 FastAPI로 나누어진 마이크로서비스 구조
    - REST API를 통해 음성 인식, 텍스트 요약 및 분류 처리

## **플로우 차트**
![플로우 차트](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject7_img01.png?raw=true)

- 로그인 창, 병원추천 시스템 창, 게시판 창, 관리자용 로그 창으로 구성

## **ERD구성**
![ERD구성](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject7_img02.png?raw=true)

- 로그, 회원가입 때 받은 정보, 게시판 관리에 필요한 board로 구성 
    - log 테이블은 관리지만 접근 가능

### **로그인 구성**

![로그인 구성](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject7_img03.png?raw=true)

- 로그인 창
- 위치추적 동의 창
- 비밀번호 찾기 창
    - 아이디, 이름, 전화번호를 통해 비밀번호 찾기
- 회원가입 창
    - 회원가입 시 필수 양식을 준수하지 않을 경우 가입 불가

### **홈페이지 구성**

![홈페이지 구성](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject7_img04.png?raw=true)

- 일반 사용자 및 비회원 
    - 위·경도 및 주소지로 위치 선택 
    - 오디오 및 텍스트로 이슈 설명 
    - 입력한 병원 수에 따른 추천 
- 관리자
    - 관리자 권한으로 접근 가능
    - 사용자들의 이용기록 확인 

### **게시판 구성**

![홈페이지 구성](https://github.com/tae2on/tae2on.github.io/blob/main/assets/img/miniproject7_img05.png?raw=true)

- 일반 사용자
    - 게시글 작성 
    - 자신이 작성한 게시글 삭제 및 수정  
- 관리자 
    - 모든 게시글 삭제 및 수정

## **고찰**
벌써 마지막 프로젝트라는 게 믿기지 않을 정도로 시간이 빠르게 흘렀던 것 같습니다. <br>
이번 프로젝트는 단순한 미니 프로젝트가 아닌 서비스를 배포까지 목표로 하며 완성도 높은 하나의 프로젝트로 다가왔습니다. <br>
로컬 환경에서는 잘 동작하던 시스템이 배포 과정에서 예기치 않은 오류를 발생하였고 이를 해결하는 과정에서 팀원들과의 협업이 중요하다는 것을 다시 한번 깨달았습니다. 특히 오류의 원인을 분석하고 해결 방안을 논의하는 과정에서 팀워크의 가치를 체감했으며 이러한 경험을 통해 문제 해결 능력과 소통 능력을 한 단계 더 성장시킬 수 있었습니다. 배포까지 완료하며 큰 재미와 성취감을 느낄 수 있게 해준 프로젝트였던 것 같습니다. 