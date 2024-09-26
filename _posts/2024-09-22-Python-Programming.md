---
title: 파이썬 프로그래밍 | Python Programming
author: tae2on
date: 2024-09-06 17:30:00 +/- 00:15
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Python, Jupyter Notebook]		# TAG는 반드시 소문자로 이루어져야함!
---
## 자료형(list, dictionary, tuple)
여러 값을 한꺼번에 저장하고 관리하기 위한 자료형

|      | 리스트 (List)                                | 딕셔너리 (Dictionary)                                         | 튜플 (Tuple)                             |
|------|----------------------------------------------|------------------------------------------------------------|------------------------------------------|
| 정의 | 순서가 있는 변경 가능한 데이터 구조               | 키와 값 쌍으로 이루어진 변경 가능한 데이터 구조                    | 변경 불가능한 데이터 구조                   |
| 선언 | 대괄호([])로 선언 <br> list() 함수를 통해 선언       | 중괄호({})로 선언,  <br> dict() 함수를 통해 선언 <br> (중괄호만 사용할 경우 집합(set) 자료형으로 인식) | 소괄호(())로 선언 <br> tuple() 함수를 통해 선언  |
| 예시 | list_a = [1, '5', 7, 9.3]                   | dic_a = {'name': 'mark', 'age': 23}                        | tup_a = (2, 3, '7')                     |

## 흐름 제어(조건문과 반복문)

### 연산자
조건에 따라 실행 흐름을 제어하거나 반복하는 구문에서 사용하는 연산자

|      | **bool 연산자**                                 | **비교 연산자**                                        | **논리 연산자**                         |
|------|-------------------------------------------------|-------------------------------------------------------|-----------------------------------------|
| **정의** | 참(True) 또는 거짓(False) 값을 반환하는 연산자      | 두 값을 비교하여 그 결과를 참 또는 거짓으로 나타내는 연산자  | 하나 이상의 조건을 결합하여 참 또는 거짓을 반환하는 연산자 |
| **종류** | True, False                                     | ==, !=, <, >, <=, >=                                  | and, or, not                           |

### 조건문
if~ elif~ else: 위에서 아래로 조건을 확인

```yaml
if (조건문1):
  코드1
elif (조건문2):
  코드2
else:
  코드3
```

while loop: 조건문이 False일때 혹은 break 만났을 때 종료 
```yaml
while 조건문:
  코드
  조건 변경문 
```
### 반복문

range(a, b, c): a부터 b전까지 c씩 증가시킨 값
for loop: 순서대로 값을 출력하며 코드를 반복 수행하며 break를 만났을 때 종료 

```yaml
while 조건문:
  코드
  조건 변경문 
```


## 함수 생성 및 활용


<!-- <div class="post-meta text-muted">
  <span>
    Posted <time>2023년 9월</time> · <time>16일</time>
  </span>
  
  <span>
    Updated <time>2024년 4월</time> <time>1일</time>
  </span>
  
  <div class="d-flex justify-content-between">
    <span> 
      By <em> <a href="https://github.com/tae2on/">tae2on</a> </em>
    </span>
    
    <span class="readtime" data-bs-toggle="tooltip" data-bs-placement="bottom" data-bs-original-title="1506 words">
      <em>8 min</em> read
    </span>
  </div>
</div>

</div> -->
