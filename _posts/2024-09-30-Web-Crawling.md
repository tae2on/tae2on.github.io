---
title: "웹 크롤링 | Web Crawling"
date: 2024-10-01 17:01:45 +0900
achieved: 2024-09-23 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook, Pandas, Web Crawling]
---
---------- 	
> KT 에이블스쿨 6기 웹 크롤링에 진행한 강의 내용 정리한 글입니다. 
{: .prompt-info } 

## **URL**
- 주소와 데이터: `?`를 기준으로 앞부분은 주소, 뒷부분은 데이터를 나타냅니다. 
- 프로토콜: `http://`와 같이 사용되며 웹에서 데이터를 주고받은 규칙을 의미합니다. 
- 경로: `main/`과 같이 리소스의 위치를 나타냅니다. 
- 포트: 서버와 클라이언트 간의 통신 포트 번호로, HTTP의 경우 일반적으로 `80`입니다. 
- 쿼리: 서버에 추가적인 데이터를 전달하는 문자열로, `mode=LSD`와 같은 형식입니다. 

## **Client & Server**
- Client: 서버에 요청을 보내는 주체로 웹 브라우저와 같은 애플리케이션을 통해 사용자와 상호작용을 합니다. 
- Request: 클라이언트는 사용자가 입력한 정보를 기반으로 서버에 데이터를 요청합니다. 이 요청은 HTTP 프로토콜을 통해 전송되며 URL을 통해 특정 리소스를 지정합니다. 
- Server: 클라이언트의 요청을 처리하고 필요한 데이터를 제공하는 시스템입니다. 
- Response: 서버는 클라이언트의 요청을 수신한 후 해당 요청에 따라 필요한 데이터를 생성하거나 검색하여 클라이언트에게 전송합니다. 

## **HTTP 요청 방법**
- Get: URL에 쿼리를 포함하여 데이터를 요청합니다. 전송 가능한 데이터 크기가 작습니다. 
- Post: 요청의 본문(body)에 쿼리를 포함하여 데이터를 전송합니다. 

## **HTTP 상태 코드**
- 2XX(Success): 요청이 성공적으로 처리되었음을 나타냅니다.
- 3XX(Redirect): 요청된 리소스의 위치가 변경되었음을 나타내며 클라이언트가 다른 URL로 이동해야함을 알립니다. 
- 4XX(Request Error): 클라이언트의 요청에 오류가 있거나 요청이 잘못된 경우를 나타냅니다. 
- 5XX(Server Error): 서버가 요청을 처리하는 도중 오류가 발생했음을 나타냅니다. 

## **Cookie & Session & Cache**
- Cookie: 클라이언트의 브라우저에 저장되는 문자열 데이터입니다. (ex. 로그인 정보, 상품 정보 등)
- Session: 클라이언트의 브라우저와 서버 간의 연결 정보를 유지합니다. 주로 서버 측에서 관리됩니다. (ex. 자동 로그인, 장바구니 정보 유지 등)
- Cache: 클라이언트와 서버의 RAM(메모리)에 저장되는 데이터로 데이터를 빠르게 입출력할 수 있도록 합니다. (ex. 웹 페이지 정적 리소스, API 응답 캐싱 등)

## **웹 언어 & 프레임워크**
**Client (Frontend)**
- HTML: 웹 페이지의 구조와 내용을 정의
- CSS: 웹 페이지의 스타일 지정 (Bootstrap, Semantic UI, Materialize, Material Design Lite)
- Javascript: 웹 페이지에 동적인 기능을 추가
<br><br>
**Server (Backend)**
- Python (Django, Flask, FastAPI)
- Java (Spring)
- Ruby (Rails)
- Scala (Play)
- JavaScript (Express (Node.js))

## **Scraping & Crawling**
- Scraping: 특정 데이터를 수집하는 작업
- Crawling: 웹서비스의 여러 페이지를 이동하며 데이터를 수집하는 작업(spider, web crawler, bot 용어 사용)

## **Internet**
컴퓨터를 연결하여 TCP/IP 프로토콜을 이용해 정보를 주고받는 글로벌 컴퓨터 네트워크
- 해저 케이블을 사용하여 전세계 컴퓨터에 접속
- 무선 인터넷은 매체(media)를 주파수 사용

## **웹 페이지의 종류**
- 정적 페이지 (Static Page): 웹 브라우져에 화면이 한번 뜨면 이벤트에 의한 화면의 변경이 없는 페이지 
- 동적 페이지 (Dynamic Page): 웹 브라우져에 화면이 뜨고 이벤트가 발생하면 서버에서 데이터를 가져와 화면을 변경하는 페이지 

## **웹 크롤링 방법**
### **Requests 이용**
받아오는 문자열에 따라 두 가지 방법으로 구분
- json 문자열로 받아서 파싱하는 방법: 주로 동적 페이지 크롤링할 때 사용
- html 문자열로 받아서 파싱하는 방법: 주로 정적 페이지 크롤링할 때 사용
- API 사용: 서로 다른 소프트웨어 시스템 간에 데이터와 기능을 요청하고 교환

### **Selenium 이용**
- 자동화를 목적으로 만들어진 다양한 브라우져와 언어를 지원하는 라이브러리
- chromedriver 사용

### **Headless**
- 브라우져를 화면에 띄우지 않고 메모리상에서만 올려서 크롤링하는 방법

## **웹 크롤링 절차**
### **동적 페이지**
- URL → 웹 페이지 분석: 개발자 도구(`Fn + F12`)을 통한 HTML 구조 파악 
- 서버에 데이터 요청: `request` 라이브러리를 통한 웹페이지 요청 후 response으로 `json(str)` 형식 데이터 수집 
- 데이터 파싱: 수집된 `json` 데이터를 `json` 라이브러리로 파싱하고 `list`, `dict` 형태로 변환 후 `DataFrame` 저장

### **정적 페이지**
- URL → 웹 페이지 분석: 개발자 도구(`Fn + F12`)을 통한 HTML 구조 파악 
- 서버에 데이터 요청: `request` 라이브러리를 통한 웹페이지 요청 후 response으로 `html(str)` 형식 데이터 수집
- 데이터 파싱: 수집된 `html` 데이터를 `BeautifulSoup` 라이브러리로 파싱하고 `list`, `dict` 형태로 변환 후 `DataFrame` 저장
