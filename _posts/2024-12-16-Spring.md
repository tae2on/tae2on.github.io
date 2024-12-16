---
title: "가상화 클라우드 | Virtualized Cloud" 
date: 2024-12-16 23:52:24 +0900
achieved: 2024-12-09 17:30:00 +0900
math: true
categories: [Bootcamp, KT Aivle School]
tags: [Bootcamp, KT Aivle School, Virtualized Cloud, Azure]
---
----------
> KT 에이블스쿨 6기 가상화 클라우드에 진행한 강의 내용 정리한 글입니다. 
{: .prompt-info } 

## **가상화 및 클라우드 개요**

### **기존 환경의 문제점과 가상화를 통한 해결방안**
**IT 에너지 사용 현황**
- 42%: 인프라 유지에 사용
- 30%: 애플리케이션 유지에 사용
- 23%: 애플리케이션 투자에 사용
- 5%: 인프라 투자에 사용

**주요 문제점**
- 극도의 시스템 복잡성
    - 유지보수에 과도한 비용이 소모되고 신규 투자는 제한적
- 빈약한 인프라 의존
    - IT운영이 안정성 유지에 치중되어 경쟁력 강화를 위한 혁신이 어려움
    - 기존 인프라가 새로운 기술을 수용하기 어려워 지속적인 개선이 불가능한 상황

**해결방안**
- 모든 IT 자산의 가상화
- 가상화 플랫폼을 이용하여 동적이고 유연한 업무 인프라를 구축
- 데이터센터의 모든 리소스를 가상화

**기존 IT 서비스 환경과 클라우드 환경 비교**
- 기존 IT 서비스 환경 (On-Premises) 
    - 혁신을 위한 시도가 적음
    - 실패의 비용이 높음
    - 혁신 속도 저하
- 클라우드 환경 (Public Cloud)
    - 혁신을 위한 시도가 많아짐
    - 실패의 비용이 낮음
    - 많은 혁신 가능

### **가상화 유형**
- 서버 가상화: 하나의 물리적 서버를 여러 가상 서버로 분할하여 운영, 서버 자원의 효율성을 극대화
- 네트워크 가상화: 물리적 네트워크를 논리적으로 분리하거나 통합하여 유연성과 확장성을 제공
- 스토리지 가상화: 물리적 스토리지를 가상화하여 여러 스토리지를 하나의 논리적 스토리지로 통합
- 데스크톱 가상화: 중앙 서버에서 가상 데스크톱 환경을 제공하여 관리 용이성과 보안을 강화

### **가상화 도입 효과**
- 물리적 서버 대수 감소
    - 서비스를 위한 물리적 서버 수를 줄임으로써 데이터센터 공간을 절약하고, 효율적인 자원 활용이 가능
- 비용 절감
    - 상면 비용 절감 
        - 데이터센터의 공간 활용이 줄어들어 시설 유지비용 절약
        - 기존 5~6개의 시스템 랙 → 1개의 랙으로 운영 가능
    - 전력 비용 절감    
        - 전력 소비를 줄여 에너지 비용 대폭 감소
        - 전력 사용량 약 1016KW 절감
    - 유지보수 비용 절감
        - 하드웨어 유지보수 비용도 감소
- 그린 IT 구현
    - 탄소 배출 절감
        - 탄소 배출량 약 2.8톤 감소 (1KW당 0.5kg의 CO₂ 배출 기준)
        -약 360그루의 소나무가 흡수할 수 있는 탄소량과 동일
    - 친환경적인 IT 운영
        - 지속 가능한 에너지 사용 및 친환경 기술 도입이 가능
- 성능 및 자원 통합 
    - CPU, 메모리, 스토리지 자원의 효율적 통합으로 IT 성능 최적화
    - 물리적 서버의 의존도를 줄이고, 가상화된 환경에서의 유연한 확장이 가능

### **가상화 정의 및 발전**
운영 체제와 물리적 하드웨어를 분리하여 IT 문제를 해결하는 기술

1. 클라이언트 하이퍼바이저
    - 높은 이용률
    -  일부 OS/App Fault Isolation
2. 서버 하이퍼바이저
    - 하이퍼바이저 
        - 시스템에서 다수의 운영 체제를 동시에 실행할 수 있게 해주는 논리적 플랫폼
        - Type 1(Native 또는 Bare-metal) 또는 Type2 (Hosted)
    - 전체 OS/App Fault Isolation
    - 가상 머신들의 캡술화
    - Hardware Independence
3. 가상 인프라
    - 중앙 집중화 관리
    - 운영중 가상 머신의 이동
    - 자동화된 비즈니스 연속성
4. 클라우드
    - Compute, Network, Storage 자원들에 대한 정책 기반 제어
    - 보안과 장애 대응 (Fault Tolerance)
    - 어플리케이션 중심


### **물리적 리소스 공유**
![img1 daumcdn](https://github.com/user-attachments/assets/fa356bf3-9d9c-4c39-90e9-1a065b7c86a9)

- CPU 리소스 공유
    - 가상 환경에서 운영 체제는 시스템의 모든 물리적 CPU 중 할당받은 CPU만을 소유한 것으로 인식
- 메모리 리소스 공유
    - 가상 환경에서 운영 체제는 시스템의 모든 물리적 메모리 중 할당 받은 메모리만을 소유한 것으로 인식
- 가상 네트워킹
    - 가상 이더넷 어댑터와 가상 스위치는 하이퍼바이저가 소프트웨어적으로 구현하여 제공

### **클라우드 컴퓨팅**
인터넷을 통해 IT 리소스(컴퓨팅 파워, 스토리지, DB 등)를 원할 때 언제든지(On-demand) 제공하고 사용한 만큼만 비용을 지불하는 서비스

#### **클라우드 컴퓨팅의 주요 기능**
간단한 명령 또는 몇 번의 클릭만으로 필요한 IT 자원 사용 가능<br>
자원의 배포와 관리는 자동화된 프로세스를 통해 수행

- 다양한 IT 자원 제공
    - CPU: 프로세싱 파워를 필요한 만큼 유연하게 조정 가능
    - Memory: 애플리케이션 및 데이터 처리에 필요한 메모리 할당
    - Storage: 데이터를 저장하고 관리할 수 있는 가상 스토리지
    - Network: 네트워크 대역폭 및 가상 네트워크 리소스 제공
    - DB (데이터베이스): 다양한 유형의 데이터베이스 지원
    - 기타(ETC): AI, 머신러닝, 빅데이터 분석 등 추가적인 고급 서비스

#### **클라우드 컴퓨팅의 이점**
- 초기 선 투자 불필요
    - 물리적 IT 인프라 구매 및 설치 비용 절감
- 저렴한 종량제 가격
    - 사용한 만큼만 비용을 지불하는 효율적인 과금 방식
- 탄력적인 운영 및 확장 가능
    - 탄력성: 변화되는 요구 사항에 맞게 신속하게 IT 리소스를 확장하거나 축소하는 기능
- 속도와 민첩성
    - 필요한 IT 자원을 빠르게 배포하여 비즈니스 목표 달성 시간 단축
- 비즈니스에만 집중 가능
    - IT 인프라 관리 부담 감소로 핵심 비즈니스에 더 많은 자원과 시간 투입 가능
- 손 쉬운 글로벌 진출 
    - 전 세계에 분산된 클라우드 인프라를 통해 손쉽게 글로벌 서비스를 제공 가능

#### **클라우드 종류**

**Public Cloud (공용 클라우드)**
- 가장 많이 사용하는 클라우드 서비스 형태
- 클라우드 서비스 제공자(Cloud Service Provider)가 소유
- 특정되지 않은 사용자에게 리소스 및 서비스를 제공
- 보안된 네트워크 연결을 통해 접근(일반적인 인터넷 사용)

**Private Cloud (사설 클라우드)**
- 클라우드 리소스를 조직에서 소유, 관리/운영
- 담당 조직이 담당 조직 데이터센터에 클라우드 환경을 구성
- 특정사용자에게만 리소스, 서비스 제공
- 보안된 네트워크를 통해 접근(사내망 사용)

**Hybrid Cloud(하이브리드 클라우드)**
- 퍼블릭 클라우드와 프라이빗 클라우드를 결합
- 사내 데이터센터를 사용하다, 클라우드로 확장하는 개념
- 프라이빗 클라우드의 단점을 퍼블릭 클라우드로 보완
- 네트워크는 VPN 또는 전용선으로 연결

#### **클라우드 컴퓨팅 3가지 서비스 모델**
- 인프라 서비스(Infrastructure as a Service)
    - 서버, 네트워킹, 데이터 저장 등의 인프라 제공
    - 손쉽게 이전하거나 기존 온프레미스 환경과 함께 사용 가능
- 플랫폼 서비스 (Platform as a Service)
    - 개발 환경 등의 플랫폼을 제공
    - 조직에서 기본 인프라를 관리하는 필요성 제거
- 소프트웨어 서비스 (Software as a Service)
    - 소프트웨어 제공
    - 서비스 제공자가 운영 및 관리

## **Azure 기본 서비스 (네트워크, 컴퓨팅, 스토리지)**
### **Azure**
Microsoft가 제공하는 퍼블릭 클라우드 플랫폼으로 다양한 리소스와 서비스를 제공하여 유연한 IT 환경 구축을 지원

- Azure 리소스 
    - Azure가 관리하는 항목 
- Azure 리소스 그룹 
    -  리소스 그룹은 수명 주기 및 보안에 따라 단일 엔터티로 관리할 수 있도록 여러 리소스를 연결하는 논리적 컨테이너
- Azure 구독
    - Azure에서 리소스를 프로비저닝하는데 사용되는 논리적인 컨테이너
- Location (지역)
    - 대기시간이 정해진 일정 경계안의 데이터센터의 집합 
- Availability Zone (가용성 영역)
    - Region 내에서 물리적으로 분리된 데이터센터
    - 장애 발생 시 데이터 손실을 막고자 가용영역간 복제를 권장
    - 왕복 대기 시간이 2ms 미만의 고성능 네트워크를 통해 연결
  
### **재해 복구 (Disaster Recovery)**
- 재해 종류
    - 외부 요인
        - 지진, 태풍, 홍수, 화재 등 자연재해
        - 테러로 인한 폭파, 전쟁, 해킹, 통신장애, 전력 공급 차단 등 인위적인 재해
    - 내부 요인
        - 시스템 결함, 기계오류, 관리 오류, 사용자의 실수 등
- 복구 목표
    - RPO(복구 지점 목표)
        - 데이터를 복구할 수 있는 마지막 시점
    - RTO(복구 시간 목표)
        - 시스템이 복구되기까지 걸리는 시간
- 데이터 복구 방안
    - 백업 
        - 특정 시점으로 데이터를 복원
        - Azure Backup 을 사용하여 복구에 사용할 수 있는 스냅샷을 생성
    - 데이터 복제
        - 여러 데이터 저장소 복제본에 실시간 또는 거의 실시간으로 라이브 데이터 복사본을 생성
        - Cross Region Replication (지역 간 복제)
            - 또 다른 보조 지역을 활용하여 재해 복구를 통해 지역적 또는 대규모 지리적 재해로부터 보호
            - 주 지역과 보조 지역은 모두 지역 쌍(Region Pair)을 형성
        - 지역 쌍(Region Pair)
            - 두 개의 데이터센터에도 영향을 미칠 정도의 큰 재해를 대비
            - 광범위하게 분산된 데이터센터를 통해 높은 가용성 보장  

### **Azure Vnet**
- Azure의 프라이빗 네트워크의 기본 구성 요소, 가상 사설 네트워크 환경
- Azure 리소스가 서로, 인터넷 및 특정 온-프레미스 네트워크와 안전하게 통신

**Vnet 사용**
- 인터넷 통신
    - VNet을 통해 인터넷을 활용한 안전한 통신 가능
- Azure 리소스 간 통신
    - Azure 내의 리소스(가상 머신, 데이터베이스 등) 간의 안전한 연결 제공
- 온-프레미스 리소스와 통신
    - 온-프레미스 네트워크와의 연결을 통해 하이브리드 클라우드 환경 구현
- 네트워크 트래픽 필터링
    - 네트워크 보안 그룹(NSG)을 활용하여 인바운드 및 아웃바운드 트래픽 제어
- 네트워크 트래픽 라우팅
    - 사용자 정의 라우팅 테이블로 네트워크 트래픽의 경로를 지정
- Azure 서비스에 대한 가상 네트워크 통합
    - Azure 서비스(예: Azure Storage, Azure SQL Database)와 VNet을 통합하여 보안을 강화

**사용할 네트워크 사설 대역**
- CIDR(Classless Inter-Domain Routing)
    - CIDR : 가장 일반적으로 활용되고 있는 IP 주소 할당 및 표기 방법
- 대역 결정 고려사항
    - 구축할 서비스의 규모
        - 네트워크 대역폭 요구사항은 서비스 규모에 따라 달라짐
    - IP 소모 개수
        - 네트워크 내에서 사용될 IP 주소의 개수를 고려하여 충분한 대역을 확보
    - 추후 확장 가능성
        - 향후 서비스 확장 시 추가적인 대역을 쉽게 확보할 수 있도록 설계
    - 타 시스템 연계 가능성
        - 기존 또는 다른 시스템과의 연계 및 통합 가능성을 고려
    - 기타(ETC)
        - 추가적인 요구사항(보안, 성능 등) 및 환경에 따른 제약 조건

### **Azure Subnet**
- 하나의 네트워크가 분할되어 나눠진 작은 네트워크
- Vnet 의 IP 대역을 적절한 단위로 분할 사용하는 방식
- 각 Subnet 도 Vnet와 동일하게 CIDR 을 이용해 IP 범위 지정
- Subnet 의 IP 대역은 반드시 Vnet 전체 대역 내에 존재해야 함
- Vnet내 Subnet 들의 IP 대역은 중복 할당 불가
- 사용중인 Subnet의 경우 CIDR 변경 불가

**Subnet 사용**
- 효율적인 IP 관리
    - IP 주소를 체계적으로 분배하여 네트워크 대역을 효율적으로 사용 가능
    - 각 서브넷은 고유한 IP 범위를 가지며, 중복되지 않도록 관리
- 브로드캐스트 사이즈 감소
    - 네트워크 트래픽을 서브넷 단위로 제한하여 브로드캐스트 트래픽 크기를 줄이고, 네트워크 성능 최적화
- 네트워크 분리를 통한 보안성 확보
    - 민감한 데이터 또는 리소스를 분리된 서브넷에 배치하여 접근을 제한하고 보안을 강화
    - 네트워크 보안 그룹(NSG)을 활용하여 각 서브넷의 트래픽을 제어 가능

### **Azure NIC**
- 가상머신과 가상네트워크 간의 상호 연결
- 가상머신에는 하나 이상의 NIC가 필수
- 가상머신의 크기(사양)에 따라 NIC를 2개 이상 포함할 수 있음

### **Azure Public IP**
Azure 리소스가 인터넷 통신을 하기 위해 사용하는 IP 주소

- 요금 정책
    - Public IP 사용 시, 사용량에 따라 요금이 청구
- 사용 가능 개수 제한
    - 구독당 사용 가능한 Public IP의 최대 개수가 제한
    - 필요 시 Azure에 요청하여 제한 개수를 증설도 가능 

**Public IP Type**
- Static (정적 IP)
    - 리소스를 생성할 때 고정된 IP 주소를 할당
    - IP 주소가 변경되지 않아, 정해진 IP가 필요한 리소스에 적합
- Dynamic (동적 IP)
    - Public IP를 리소스에 연결할 때 IP 주소를 할당
    - 리소스를 제거하거나 다시 시작할 경우, 할당된 IP가 변경될 수 있음

### **VM**
물리 OS에서 만들어내는 또다른 가상 운영체제 컴퓨터

**Azure VM**
Azure에서 제공하는 VM 서비스

- Azure VM은 클라우드에서 물리 서버와 동일한 기능을 수행  
    - 확장성과 유연성을 제공 
- 개발, 테스트, 프로덕션 환경 등 다양한 목적에 맞게 활용 가능
- 목적에 따른 다양한 시리즈를 제공
    - 개발 테스트용 엔트리 수준 VM
    - 경제적 버스트 가능 VM
    - 범용 컴퓨팅
    - 메모리 내 애플리케이션
    - 컴퓨팅 최적화
    - 메모리 및 스토리지
        - 스토리지: 컴퓨터, 서버에 데이터를 저장하는 저장소 역할을 수행하는 부품
            - 클라우드 스토리지: 데이터를 인터넷 또는 다른 네트워크를 통해 타사에서 유지 관리하는 오프사이트 스토리지 시스템으로 전송하여 저장할 수 있는 서비스
    - 고성능 컴퓨팅 가상 머신

### **네트워크 보안그룹 (NSG)** 
Azure 리소스와 주고받는 네트워크 트래픽을 필터링

- 규칙 설정 항목 
    - 소스(Source)
        - 트래픽의 소스 지정: Any, 특정 IP 주소, Service Tag, Application Security Group
    - 소스 포트 범위(Source Port Ranges)
        - 네트워크 트래픽의 소스 포트를 지정: Any, 특정 포트(예: 8080) 또는 범위 지정
    - 목적(Destination)
        - 트래픽의 목적지 설정: Any, 특정 IP 주소, Service Tag, Application Security Group
    - 목적 포트 범위(Destination Port Ranges)
        - 트래픽의 대상 포트를 지정: HTTP, HTTPS, SSH, RDP 등
    - 프로토콜(Protocol)
        - 통신에 사용되는 프로토콜 지정: TCP, UDP, ICMP, ESP, AH 또는 Any
    - 작업(Action)
        - 트래픽의 허용 여부를 지정: 허용(Allow) 또는 거부(Deny)
    - 우선 순위(Priority)
        - NSG 규칙의 처리 우선 순위를 설정
        - 낮은 숫자가 높은 우선 순위를 가짐
    - 이름(Name)
        - 안 규칙의 고유 이름 설정

### **애플리케이션 보안 그룹 (ASG)** 
- 네트워크 보안 정책을 간소화하고 관리하기 위한 기능
- 네트워크 보안그룹 규칙을 설정할 때 특정 IP주소, 서브넷 대신 논리적인 그룹 지정 가능

### **Storage Account**
Azure의 다양한 데이터 저장 서비스를 관리하고 액세스하는 기본 단위

**주요서비스**
- Disk Storage: VM에 연결된 디스크
- File Storage: 클라우드 네트워크 공유디스크
- Blob Storage: 대용량 비정형 데이터 저장
- Queue Storage: 메시지 큐를 통해 애플리케이션 간 비동기 메시징 지원
- Table Storage: 비관계형 데이터 저장

### **Azure 주요 스토리지 종류**
- Azure Disk: Azure 가상 머신(VM)에서 사용되는 고성능 블록 스토리지 
- Azure Files: 완전 관리형 파일 공유 서비스로, SMB(Windows) 및 NFS(Linux) 프로토콜 지원
- Blob Storage: 대규모 비정형 데이터(이미지, 동영상, 로그 등)를 저장하는 오브젝트 스토리지 서비스

#### **Azure Disk**
Azure Virtual Machines와 함께 사용되는 블록 수준 스토리지 볼륨

**사용 가능한 디스크 유형**
- Ultra 디스크
- 프리미엄 SSD
- 표준 SSD 및 표준 HDD

**Managed Disk 의 이점**
- 뛰어난 내구성 및 가용성
    - 안정적인 데이터 보호와 높은 가용성 보장
- 확장성, 간편함
    - 워크로드에 따라 디스크 크기 및 성능 확장 가능
- Azure Backup 지원
    - 데이터 백업 및 복구를 간편하게 수행 가능
- 세부적인 액세스 지원
    - 데이터 관리와 보안 설정을 위한 세부적인 제어 제공

#### **Azure files**
SMB(서버 메시지 블록) 프로토콜 또는 NFS(네트워크 파일 시스템) 프로토콜을 통해 액세스할 수 있는, 클라우드에서 완전 관리형 파일 공유를 제공

**주요 이점**
- 공유 액세스
    - 여러 클라이언트에서 동시에 파일에 액세스 가능
- 완벽한 관리
    - 클라우드에서 자동으로 파일 공유를 관리하여 유지보수 부담 감소
- 스크립팅 및 도구 지원
    - Azure CLI, PowerShell 등 다양한 스크립팅 도구와의 호환성 제공
- 복원력
    - Azure의 내구성과 안정성을 바탕으로 데이터 보호 및 가용성 보장
- 친숙한 프로그래밍
    - 기존 파일 공유 프로토콜과 동일한 방식으로 작업 가능, 학습 곡선이 낮음

#### **Azure Blob Storage**
클라우드를 위한 Microsoft의 개체 스토리지 솔루션

**설계 목적**
- 이미지 및 문서 제공
    - 브라우저에 이미지나 문서를 직접 제공
- 파일 저장
    - 분산된 사용자를 위한 파일 저장소 역할 수행
- 비디오 및 오디오 스트리밍
    - 비디오 및 오디오 콘텐츠의 원활한 스트리밍 지원
- 로그 파일 관리
    - 로그 데이터를 저장하고 분석을 위해 쓰기 작업 지원
- 백업 & 복원 및 보관
    - 백업/복원, 재해 복구 및 데이터 아카이빙 용도로 적합
- 데이터 저장
    - 온프레미스 또는 Azure 호스팅 서비스에서 분석 목적으로 데이터 저장

## **Azure 고가용성 서비스(로드밸런서, 오토스케일링)**
### **Load Balancer (부하 분산 장치)**
- 외부에서 들어오는 트래픽을 특정 알고리즘을 기반으로 다수의 서버로 분산
- 서버 문제를 자동 감지하고 사용 가능한 서버로 전달하여 해당 서버들이 제공하는 서비스 안정성을 높임

**Azure Load Balancer**
OSI 모델의 4계층(L4)에서 작동하는 트래픽 분산 서비스

**백엔드 풀 인스턴스**
- 가상 머신(VM)
- 가상 머신 스케일 세트(VMSS)의 인스턴스 
    - VMSS(Virtual Machine Scale Set)
        - 부하 분산된 VM의 그룹
        - VM 수는 정의된 일정에 따라 자동 확장 및 축소
    - VMSS 주요 이점
        - 손쉬운 여러 VM 만들기 및 관리
        - 고가용성 및 애플리케이션 복원력 제공
        - 리소스 수요 변화에 따라 자동으로 애플리케이션 크기 조정
        - 대규모 작업 

**유형**
- 공용 부하 분산 장치(Public Load Balancer)
    - 가상 네트워크 내의 VM에 대해 아웃바운드 연결을 제공
    - 공용 IP 주소를 통해 인터넷과 통신 가능
- 내부 부하 분산 장치 (Internal Load Balancer)
    - 사설 IP를 사용하는 경우 적합
    - 프런트 엔드에서만 사설 IP가 필요한 트래픽 분산 지원

#### **사용 시나리오**
1. 내/외부 트래픽을 가상머신으로 부하 분산
2. 리전 내 리소스에 대한 가용성 상승
3. 가상머신에 대한 아웃바운드 연결
4. 상태프로브를 사용하여 부하가 분산된 리소스 모니터링

#### **구성요소**

**프런트 엔드 IP**
Azure Load Balancer의 IP 주소로, 클라이언트의 접점을 의미
- 공용 IP 주소
- 개인 IP 주소

**백 엔드 풀**
들어오는 요청을 처리하는 가상 머신(VM) 또는 가상 머신 확장 집합(VMSS) 인스턴스들의 그룹

**상태 프로브**
백 엔드 풀의 인스턴스 상태를 확인하는 데 사용

**부하 분산 장치 규칙**
백 엔드 풀 내의 모든 인스턴스에 들어오는 트래픽이 배포되는 방법을 정의

### **로드밸런서 알고리즘**
부하를 백엔드 노드에게 전달하는 규칙

- 알고리즘
    - 라운드 로빈: 순차적으로 서버에 트래픽을 할당하여 분산
    - 가중치 라운드 로빈: 가중치를 설정하고 가장 높은 가중치가 설정된 서버부터 트래픽 처리
    - 최소 연결 방식: 가장 처리를 적게 한 서버에게 우선적으로 할당하여 분산하는 방식
    - IP 해시 방식: 클라이언트의 IP 주소를 기반으로 동일한 IP의 트래픽은 항상 동일한 서버로 연결
    - URL 스위칭 방식: 특정 하위 URL 들은 특정 서버로 처리하는 방식
    - 컨텍스트 스위칭 방식: 특정 리소스 요청에 대해 특정 서버를 연결
    - Azure 로드밸런서 알고리즘
        - 5-튜플 해시 알고리즘
        - 패킷의 정보 5가지를 기반으로 분배하는 방식
            - 소스 IP (Source IP): 패킷을 보낸 클라이언트의 IP 주소
            - 목적지 IP (Destination IP): 패킷이 도달할 서버의 IP 주소
            - 소스 포트 (Source Port): 클라이언트 애플리케이션에서 사용하는 포트 번호
            - 목적지 포트 (Destination Port): 서버 애플리케이션에서 사용하는 포트 번호
            - 프로토콜 (Protocol): TCP/UDP 같은 전송 프로토콜의 종류

### **Horizontal & Vertical Scaling**
**Horizontal Scaling(수평적 크기조정)**
- 스케일 아웃: 가상머신을 추가하는 것
- 스케일 아웃 : 가상머신을 제거하는 것

**Vertical Scaling(수직적 크기조정)**
- 스케일 업 : 메모리, CPU 등 리소스의 용량을 증설
- 스케일 다운 : 리소스 용량을 축소

## **Docker & Container**
### History of Deployment
1. Bare Metal 방식
- 하드웨어에 OS를 설치하고 애플리케이션을 구성
- 보통 하나의 하드웨어에 하나의 애플리케이션으로 구성
- 추가 애플리케이션 필요시 하드웨어 구입, 파워 및 네트워크 연결 등 구동에 필요한 설정, 애플리케이션 설치 및 구성 단계를 거침
2. Virtualized 방식
- 하드웨어에 OS 대신 Hypervisor 를 설치
    - Hypervisor: 물리적 리소스를 논리적으로 분리시켜 VM에 할당하여 VM을 생성하는 소프트웨어
    - VM(가상머신): 물리적 리소스를 논리적으로 분리시켜 생성한 가상 컴퓨팅 환경
- Hypervisor를 통해 각 VM을 필요한 만큼 생성
- VM에 OS를 설치(Guest OS라고 표현) 하고 필요한 애플리케이션을 구성
- 추가 애플리케이션 필요시 VM을 추가하고 Guest OS 설치 후 애플리케이션 구성
3. Containerized 방식
- 하드웨어에 OS를 설치하고, Container Engine 을 설치
    - Container Engine : 물리적인 리소스를 논리적으로 분리시켜 컨테이너를 생성하는 소프트웨어
    - Container : 논리적인 구획인 컨테이너를 만들고 애플리케이션을 설치하여 서버처럼 사용하는 패키지
- Container Engine을 통해 필요한 애플리케이션을 구동할 Container를 생성
- 추가 애플리케이션 필요시 애플리케이션을 구동할 Container를 생성

### **Docker**
오픈소스 기반 Container Engine (컨테이너를 구동해주는 소프트웨어)

- 사용 가능한 OS
    - Linux
    - Windows
    - Mac

#### **Dockerfile**
컨테이너 이미지를 생성하기 위한 레시피 파일
- 이 파일에 이미지 생성과정을 문법에 따라 작성하여 저장
- FROM, WORKDIR, RUN, CMD 등 용도에 따른 명령어 모음

```Dockerfile
FROM ubuntu:18.04                   # 베이스 이미지
RUN apt-get update && apt-get install -y vim apache2  # 패키지 설치
COPY index.html /var/www/html/      # 파일 복사
CMD ["/usr/sbin/apachectl", "-D", "FOREGROUND"]  # 컨테이너 실행 시 명령어
LABEL version="1.0"                 # 메타데이터 설정
ENV USER=docker                     # 환경 변수 설정
EXPOSE 80                           # 포트 80 노출
USER docker:docker                  # 실행 사용자 설정
WORKDIR /home/docker                # 작업 디렉토리 설정
```

#### **Docker HUB**
수많은 컨테이너 이미지들을 서버에 저장하고 관리
- Rate limit 사항
    - 익명 유저: ip 기반으로 6시간동안 100번 제한
    - 로그인 유저: 계정을 기반으로 6시간동안 200번 제한

#### **Docker 계정 로그인**

```bash
# 로그인 상태 확인인
docker info | grep Username
# 도커 로그인 명령
docker login
# 도커 로그아웃 명령
docker logout
```

#### **Docker 이미지 생성 & 업로드 & 컨테이너 실행**
**이미지 생성 (빌드)**
- 도커 파일(Dockerfile)을 작성 
- Dockerfile 이 있는 경로에서 빌드 명령 수행

```bash
# 이미지 빌드 명령어 
# docker build -t <도커허브 계정명>/<빌드할 이미지명>:<태그명> .
docker build -t aivleaccount/myapp:v5 .
```

**이미지 업로드 (Push)**
- 생성한 이미지를 내 계정 도커허브 저장소에 업로드

```bash
# 이미지 업로드 명령어
# docker push <도커허브 계정명>/<이미지명>:<태그명>
docker push aivleaccount/myapp:v5
```

**Docker 이미지를 통해 컨테이너 실행**
```bash
# 컨테이너 실행 명령어
# docker run -d -p <로컬 포트>:<컨테이너 포트> --name <컨테이너 이름> <이미지명>:<태그명>
docker run -d -p 8080:80 --name myapp-container aivleaccount/myapp:v5
```

### **Container Image**
컨테이너를 실행하는데 필요한 프로그램, 소스코드 등을 묶어 놓은 소프트웨어 패키지
- 하나의 이미지로 여러 컨테이너를 생성 가능
- 이미지는 컨테이너가 삭제되더라도 삭제되지 않음(직접 삭제 필요)
- 이미지 사용
    - 이미지는 Dockerfile을 사용하여 생성 (Build cmd)
    - 이미지를 사용하여 Container 실행 (Run cmd)

### **Image Name** 
- Image Naming 방식: `<Namespace>`/`<ImageName>`:`<Tag>`
- namespace: 이미지가 저장되어 있는 저장소
- Image Name: 이미지 이름
- Tag: 이미지를 관리하는 사용자가 지정한 태그 또는 버전

### **Private Image**
필요에 따라 이미지를 공개하지 않고 임의의 기업, 부서, 개인만 사용하도록 구성
- Public 저장소가 아닌 Private 저장소를 구성하여 사용
- 저장소서버의 주소 및 포트 설정을 하고, Namespace 로 지정
- 이미지를 커스터마이징 가능

### **ACI & ACR**
**ACI**
- Azure에서 컨테이너를 실행해주는 서비스
- 사용가능한 이미지 크기 : 최대 15GB
- 호환되는 Container Registry : ACR, DockerHub 및 타사 Registry

**ACR**
- 관리형 레지스트리
- 컨테이너 이미지와 관련 아티팩트를 저장하고 관리

## **Kubernetes 개요 및 주요 아키텍쳐**
### **Container Orchestration**
- 다수의 컨테이너를, 다수의 시스템에서, 각각의 목적에 따라, 배포, 복제, 장애복구 등 총괄적으로 관리
- 수천, 수만개의 컨테이너를 한정된 관리자가 손수 관리하는 것은 사실상 불가
- Container Orchestration을 해주는 도구
    - Marathon: Apache Mesos 기반의 오케스트레이션 도구
    - Microsoft Azure Service Fabric: 마이크로소프트의 Azure 플랫폼에서 제공하는 컨테이너 오케스트레이션 서비스
    - Nomad: HashiCorp에서 제공하는 오케스트레이션 도구로 단순하고 유연한 설계가 특징
    - AWS ECS (Elastic Container Service): AWS 클라우드 환경에서 제공하는 컨테이너 관리 및 오케스트레이션 서비스
    - Docker Swarm: Docker에서 제공하는 네이티브 오케스트레이션 도구, 컨테이너 배포 및 클러스터 관리에 사용

### **Kubernetes**
컨테이너형 애플리케이션의 배포, 확장, 관리를 자동화하는 오픈 소스 Orchestration 시스템

#### **아키텍쳐**
- Cluster
    - 컨테이너 형태의 애플리케이션을 관리하는 물리 또는 가상 환경의 노드들의 집합
    - Master + Worker Node 집합
        - Master Node: 관리를 위한 제어 노드
            - kube-api-server
                - API Server: Master Node의 중심에서 모든 클라이언트와 구성요소로부터의 요청을 받아 처리 
            - kube-controller-manager
                - Controller Manager: 클러스터의 상태를 조절 하는 컨트롤러들을 생성, 배포
            - kube-etcd
                - ETCD: 클러스터 내 모든 구성 데이터를 저장하는 저장소
            - kube-scheduler
                - Scheduler: 파드의 생성요청 시 해당 파드를 위치시킬 노드 선정
        - Worker Node: 컨테이너가 배포될 물리 또는 가상머신
            - kubelet
                - 클러스터 내 각 노드에서 실행되는 에이전트
                - Pod에서 컨테이너가 확실하게 동작하도록 관리
            - kube-proxy
                - 클러스터 내 각 노드에서 실행되는 네트워크 프록시
                - Service 개념의 구현부
            - Container Runtime
                - Container 를 배포하고 이미지를 관리
                - Docker Engine
                - Containerd
                - CRIO
- Addons
    - 추가 설치를 통해 Kubernetes의 기능을 확장 시킬 수 있는 도구들
        - Dashboard
        - Monitoring
        - Logging
- AKS 클러스터
    - Control Plane: Azure가 관리
    - Node: 고객이 관리

## **컨테이너 배포, 통신, 볼륨 관리** 
### **Kubernetes Object**
Kubernetes 의 상태를 나타내기 위해 사용하는 오브젝트
- 기본 오브젝트
    - Pod: 컨테이너를 실행하는 가장 작은 배포 단위
    - Service: 네트워크를 통해 Pod를 외부와 연결하는 역할
    - Namespace: 클러스터 내 리소스를 격리하고 관리하는 논리적 그룹
- 오브젝트의 두 필드
    - Spec: 사용자(운영자)가 원하는 상태를 지정하는 필드
    - Status: 클러스터 내 오브젝트의 현재 상태를 출력하는 필드

### **Kubernetes Controller Object**
클러스터의 상태를 관찰하고 필요한 경우에 오브젝트를 생성, 변경을 요청하는 오브젝트
- 컨트롤러는 클러스터의 현재 상태를 정의된 상태에 가깝게 유지하려는 목적
- 컨트롤러 유형
    - Deployment: 애플리케이션 배포 및 업데이트를 관리하는 컨트롤러
    - Replicaset: Pod의 개수를 관리하여 원하는 복제본 수를 유지
    - Daemonset: 각 노드마다 하나의 Pod를 실행하도록 보장
    - Job: 일회성 작업을 실행하고 완료되면 종료
    - CronJob: 스케줄에 따라 주기적으로 작업을 실행하는 컨트롤러

### **YAML**
Kubernetes 에서 오브젝트를 기술하는 포맷

- json 형식으로도 사용가능
- 필드값
    - apiVersion: 해당 오브젝트를 생성하기 위해 사용하고 있는 k8s api version
    - kind: 오브젝트의 유형
    - metadata: 오브젝트를 구분할 데이터(name, label, namespace 등)
    - spec: 오브젝트의 원하는 상태

### **kubectl**
Kubernetes에 명령을 내리는 CLI(Command Line Interface)
- kubectl 명령어 구조
- 오브젝트와 컨트롤러를 생성, 수정, 삭제
    - COMMAND: 실행할 명령어 (예: get, create, delete 등)
    - TYPE: 대상 오브젝트 유형 (예: pod, service, deployment 등)
    - NAME: 오브젝트 이름
    - FLAGS: 추가 옵션 (예: -o json, --namespace 등)

```bash
# kubectl 명령 구조 
kubectl [COMMAND] [TYPE] [NAME] [FLAGS]
```

### **Pod**
- Kubernetes의 가장 작은, 최소 단위 Object
- 하나 이상의 컨테이너 그룹
- 네트워크와 볼륨을 공유

#### **Kubectl**
```bash
# yaml 파일을 사용하여 Pod를 생성 하는 명령어
# kubectl create -f <yaml 파일명>
kubectl create -f pod.yaml

# kubectl 명령으로 Pod를 생성하는 명령어
# kubectl run <pod명> --image=<이미지명:버전> --port=<포트번호>
kubectl run pod1 --image=nginx:1.14.0 --port=80
```

### **Namespace**
단일 물리 클러스터에서 리소스를 논리적으로 격리하여 그룹화하는 오브젝트<br>
k8s 클러스터가 배포되면 기본 Default라는 Namespace 생성
- Namespace를 사용하는 목적
    - 사용자들을 그룹화
    - 배포레벨에 따라 그룹화

#### **기본 Namespace**
- default: 다른 네임스페이스가 없는 오브젝트를 위한 기본 네임스페이스
- kube-system: k8s 시스템에서 생성한 오브젝트를 위한 네임스페이스
- kube-public: 모든 사용자(인증되지 않은 사용자 포함)가 읽기 권한으로 접근 가능
한 네임스페이스
- kube-node-lease: 노드의 HeartBeat 체크를 위한 네임스페이스

#### **Kubectl**

```bash
# yaml 파일을 사용하여 Namespace를 생성 하는 명령어
# kubectl create -f <yaml 파일명>
kubectl create -f ns1.yaml

# kubectl 명령으로 Namespace를 생성하는 명령어
# kubectl create namespace <Namespace 명>
kubectl create namespace ns1
```

#### **지정하여 명령 수행 - Kubectl**

```bash
# pod.yaml 파일을 사용하여 ns1 이라는 namespace에 pod를 생성 하는 명령어
# kubectl create -f <yaml 파일명> -n <namespace 명>
kubectl create -f pod.yaml -n ns1

# ns1 namespace에 생성된 Pod를 조회하는 명령어
# kubectl get pod -n <namespace 명>
kubectl get pod -n ns1
```

### **Kubernetes Controller**
**ReplicaSet**
- ReplicaSet은 Pod의 개수를 유지
- yaml을 작성할 때 replica 개수를 지정하면 그 개수에 따라 유지

**Deployment**
- Deployment는 ReplicaSet을 관리하며 애플리케이션의 배포를 더욱 세밀하게 관리
- 초기 배포 이후에 버전 업데이트, 이전 버전으로 Rollback도 가능

**Deployment Update**
- 운영중인 서비스의 업데이트시 재배포를 관리
- 2가지 재배포 방식
    - Recreate
        - 현재 운영중인 Pod들을 삭제한 후 업데이트 된 Pod들을 생성
        - Downtime이 발생하기 때문에 실시간 서비스에는 권장하지 않는 방식
    - Rolling Update
        - 업데이트 된 Pod를 하나 생성하고 구버전의 Pod를 삭제 
        - Downtime 없이 업데이트가 가능

**Deployment Rollback**
- Deployment는 이전버전의 ReplicaSet을 10개까지 저장 
    - revisionHistoryLimit 속성을 설정하면 개수를 변경 가능
- 저장된 이전 버전의 ReplicaSet을 활용하여 Rollback

### **Deployment - Kubectl**

```bash
# yaml 파일을 사용하여 생성 하는 명령어
# kubectl create -f <yaml 파일명>
kubectl create -f deployment.yaml

# kubectl 명령으로 생성하는 명령어
# kubectl create deployment <이름> \
# --image=<이미지명:버전> \
# --replicas=<Pod수>
kubectl create deployment dp \
--image=nginx:1.14.0 \
--replicas=3

# Deployment로 생성된 Pod 수를 조정(Scale)하는 명령어
# kubectl scale deployment/<Deployment명> --replicas=<조정할 Pod 수>
kubectl scale deployment/dp --replicas=3

# ReplicaSet으로 생성된 Pod 수를 조정(Scale)하는 명령어
# kubectl scale rs/<ReplicaSet명> --replicas=<조정할 Pod 수>
kubectl scale rs/rs --replicas=3
```

### **Kubernetes Object - Service**
- Pod에 접근하기 위해 사용하는 Object
- Kubernetes 외부 또는 내부에서 Pod에 접근할 때 필요
- 고정된 주소를 이용하여 접근이 가능
- Pod에서 실행중인 애플리케이션을 네트워크 서비스로 노출시키는 Object

#### **Service의 구성요소** 
- Label
    - Pod와 같은 Object에 첨부된 키와 값 쌍
- Selector
    - 특정 Label 값을 찾아 해당하는 Object만 관리할 수 있게 연결
- annotation
    - Object를 식별하고 선택하는 데에는 사용되지 않으나 참조할 만한 내용들을 Label처럼 첨부


#### **Service 유형**
- ClusterIP(default)
    - Service가 기본적으로 갖고있는 ClusterIP를 활용하는 방식      
        ```bash
        # ClusterIP 유형의 Service를 생성하는 명령어
        # kubectl create service clusterip <Service명> --tcp=<포트:타켓포트>
        kubectl create service clusterip clip --tcp=8080:80
        # ClusterIP 유형의 Service를 nginx라는 Deployment와 연결하여 생성하는 명령어
        kubectl expose deployment nginx \
        --port=8080 \
        --target-port=80 \
        --type=ClusterIP
        ```
- NodePort
    - 모든 Node에 Port를 할당하여 접근하는 방식
        ```bash
        # NodePort 유형의 Service를 생성하는 명령어
        # kubectl create service nodeport <Service명> --tcp=<포트:타켓포트>    
        kubectl create service nodeport np --tcp=8080:80
        # NodePort 유형의 Service를 nginx라는 Deployment와 연결하여 생성하는 명령어
        kubectl expose deployment nginx \
        --port=8080 \
        --target-port=80 \
        --type=NodePort
        ```
- Load Balancer
    - Load Balancer Plugin 을 설치하여 접근하는 방식
    ```bash
    # LoadBalancer 유형의 Service를 생성하는 명령어
    # kubectl create service loadbalancer <Service명> --tcp=<포트:타켓포트>
    kubectl create service loadbalancer lb --tcp=5678:8080
    # LoadBalancer 유형의 Service를 nginx라는 Deployment와 연결하여 생성하는 명령어
    kubectl expose deployment nginx \
    --port=8080 \
    --target-port=80 \
    --type=LoadBalancer
    ```

### **Kubernetes DNS**
- Kubernets는 Pod와 Service에 DNS 레코드를 생성
- IP 대신 이 DNS를 활용하여 접근 가능

### **FQDN 구성**

| 리소스 | Namespace | FQDN 구성 |
|--------|-----------|----------|
| Service | `ns1` | `svc1.ns1.svc.cluster.local` |
| Pod | `ns1` | `10-10-10-10.ns1.pod.cluster.local` |

### **Volume**
Kubernetes에서 Volume은 Pod 컨테이너에서 접근할 수 있는 데이터 저장소
- 유형
    - EmptyDir
        - Pod가 생성될 때 함께 생성되고 Pod가 삭제될 때 함께 삭제되는 임시 Volume
    - HostPath
        - 호스트 노드의 경로를 Pod에 마운트하여 함께 사용하는 유형의 Volume
    - PV/PVC
        - PV
            - Persistent Volume
            - 클러스터 내부에서 Object처럼 관리 가능 (Pod 와는 별도로 관리)
            - PV로 사용할 수 있는 Volume 유형: CephFS, csi, FC, hostpath, icsci, local, nfs 등
        - PVC
            - PersistentVolumeClaim
            - PV에 하는 요청: 사용하고 싶은 용량, 읽기/쓰기 모드 설정
            - Pod와 PV를 분리시켜 다양한 스토리지를 사용할 수 있게 구현
        - PV & PVC의 생명주기 
            - Provisioning: PV를 생성하는 단계
            - Binding: PV와 PVC를 연결하는 단계
            - Using: Pod와 PVC가 연결되어 Pod가 볼륨을 사용중인 단계
            - Reclaim: 사용이 끝난 PVC를 삭제하고 PV를 초기화

### **StorageClass**
여러가지 스토리지 유형을 표현하는 오브젝트 <br>
PV 를 동적으로 프로비저닝할 때 스토리지 지정 필요
오브젝트 필드
- provisioner
    - PV 비프로비저닝에 사용되는 플러그인 지정 필드
    - Azure의 file 서비스를 비롯하여 다양한 플러그인 사용 가능
- parameters
    - 볼륨을 설명하는 속성 값 필드
    - 최대 512개의 속성 정의 가능
    - parameters의 최대 길이는 256 KiB
- reclaimPolicy
    - StorageClass에 의해 동적으로 생성되는 PV의 reclaimPolicy 설정
    - 설정 옵션
        - Delete : pv가 삭제되면 연결된 플러그인의 스토리지 리소스 또한 함께 삭제
        - Retain : pv가 삭제되면 연결된 플러그인의 스토리지 리소스와는 연결 해제, 따라서 수동 삭제 필요
    - 생성시 지정하지 않으면 기본값 Delete
- allowVolumeExpansion
    - PV의 최초 생성 이후 확장가능 하도록 설정
    - 기본값 : true
    - 확장 이후에는 축소 불가
- volumeBindingMode
    - 볼륨과의 연결과 동적으로 PV를 생성하는 시점 제어 필드
    - 설정 옵션
        - Immediate: PVC가 생성되면 연결 및 PV 가 즉시 생성
        - WaitForFirstConsumer: PVC를 사용하는 Pod가 생성 되면 연결 및 PV 생성
        - 설정하지 않으면 기본값 : Immediate

## **컨테이너 및 노드 오토스케일링** 
### **Metric-Server**
- 리소스 메트릭을 수집하여 HPA, VPA 에서 사용하기위해 쿠버네티스 API-server에
노출
- 리눅스의 top 명령어를 Kubernetes에서 kubectl을 통해 사용

### **HPA & VPA**
- Horizontal Pod Autoscaler
    - 부하 증감에 따라 Pod의 개수를 조절해주는 리소스
- Vertical Pod Autoscaler
    - 부하 증감에 따라 Pod의 사양을 조절해주는 리소스

### **AKS 클러스터 Autoscaling**
부하 증감에 따라 Node의 개수를 확장/축소