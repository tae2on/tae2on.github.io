--- 
title: "Verilog 기반 디지털 시계 시스템"
date: 2022-06-18 15:16:45 +0900
math: true
categories: [Project]
tags: [Project, FPGA, Verilog]
---
---------- 	
> Verilog와 FPGA를 활용하여 디지털 시계 시스템 프로젝트
{: .prompt-info } 

## **개요**


### **주제선정**
기본 과제는 디지털 시계를 제작하고, 여기에 팀원 각자의 개인 기능을 추가하는 것이었다. 이를 위해 공통적으로는 LCD에 날짜와 요일, 시간을 표시하는 기본 시계를 구현하였으며, 팀원별로는 각각 차별화된 기능을 담당하였다. 구체적으로는 날짜·요일·시간을 직접 설정할 수 있는 기능, 알람이 울리면 UART를 통해 메시지를 송신하는 기능, UART 수신을 통해 모드를 변경하는 기능을 추가하였다. 이 가운데 나는 시계의 날짜, 요일, 시간을 설정할 수 있는 기능을 설계하여 구현하였다.

## **설계과정**
### **블록 다이어그램** 
![블록 다이어그램](https://github.com/tae2on/Verilog_clock_system/blob/main/img/%EB%B8%94%EB%A1%9D%EC%84%A0%EB%8F%84.png?raw=true)

### **디지털 시계**
입력 클록을 1Hz로 분주하여 초 단위로 시간이 증가하도록 설계하였습니다. 이 1Hz 펄스가 발생할 때마다 `casez` 문을 이용하여 조건을 검사하고, 초 → 분 → 시 → 일 → 월 → 연도로 자연스럽게 값이 넘어가도록 설계하였습니다. 즉, 초가 59에서 60이 되면 0으로 초기화가 되고 분이 1 증가하며, 같은 방식으로 분·시·일이 모두 경계 조건을 만족하면 상위 단위가 갱신되도록 처리하였습니다. 

```Verilog
if (clk1sec == 1) begin
				casez ({year, month, day, hour, min, sec})
					{14'd9999, 8'd12, max_date, 8'd23, 8'd59, 8'd59} : begin
						year		<=	1'd1;
						month 		<= 	1'd1;
						day		<= 	1'd1;
						hour		<=	1'd0;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'd12, 8'd1, 8'd23, 8'd59, 8'd59} : begin
						year		<= 	year + 1'd1;
						month 		<= 	1'd1;
						day		<= 	1'd1;
						hour		<=	1'd0;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'd12, max_date, 8'd23, 8'd59, 8'd59} : begin
						year		<= 	year + 1'd1;
						month 		<= 	1'd1;
						day		<= 	1'd1;
						hour		<=	1'd0;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'dz, max_date, 8'd23, 8'd59, 8'd59} : begin
						year		<= 	year;
						month 		<= 	month + 1'd1;
						day		<= 	1'd1;
						hour		<=	1'd0;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'dz, 8'dz, 8'd23, 8'd59, 8'd59} : begin
						year		<= 	year; 
						month 		<=	month;
						day		<= 	day + 1'd1;
						hour		<=	1'd0;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'dz, 8'dz, 8'dz, 8'd59, 8'd59} : begin
						year		<= 	year; 
						month 		<= 	month;
						day		<= 	day;
						hour		<=	hour + 1'd1;
						min		<=	1'd0;
						sec		<=	1'd0;
					end
					
					{14'dz, 8'dz, 8'dz, 8'dz, 8'dz, 8'd59} : begin
						year		<=	year; 
						month	 	<=	month;
						day		<=	day;
						hour		<=	hour;
						min		<=	min + 1'd1;
						sec		<=	1'd0;
					end
					
					default : begin
						year		<=	year; 
						month 		<=	month;
						day		<=	day;
						hour		<=	hour;
						min		<=	min;
						sec		<=	sec + 1'd1;
					end
				endcase
```

날짜 계산에서는 윤년 여부와 월별 최대 일수를 고려하여 설계하였습니다. 윤년의 조건인 4의 배수이면서 100의 배수가 아니거나, 400의 배수인 경우라는 규칙에 따라 판별하였습니다. 이 로직을 통해 2월이 평년에는 28일, 윤년에는 29일까지 표시되도록 구현하였습니다. 또한 1, 3, 5, 7, 8, 10, 12월은 31일, 4, 6, 9, 11월은 30일로 설정하여 실제 달력과 동일하게 동작하도록 하였습니다. 요일은 누적 일수를 기준으로 7로 나눈 나머지를 이용하여 계산하였고, LCD에는 해당 요일의 영문 약자와 함께 표시되도록 하였습니다. 

```Verilog
always @ (*) begin
	if ( (((year % 4) == 0 && (year % 100) != 0) || (year % 400) == 0)  &&	month	>	3)
		week <= ((((year-1)*365) + (((year-1)/4) - ((year-1)/100) + ((year-1)/400))) + sum_day + 1 + day) % 7;
	else
		week <= ((((year-1)*365) + (((year-1)/4) - ((year-1)/100) + ((year-1)/400))) + sum_day + day) % 7;
	end
```

### **날짜·요일·시간 설정**
사용자가 직접 연·월·일·시·분·초를 설정할 수 있는 모드를 구현하였습니다. LCD에는 현재 선택된 항목이 깜박임으로 표시되게 하였습니다. 스위치를 이용하여 항목을 이동하거나 값을 증가 혹은 감소시킬 수 있고, 윤년 및 월별 최대 일수를 반영하여 잘못된 날짜가 입력되지 않도록 제한하였습니다. 마지막 화살표 위치에서 저장을 누르면 새로운 시간 값이 시계에 적용되도록 설계하였습니다. 

```Verilog
if(sw_in == 4'b1000 && location < 3'd6) 
    location <= location + 1;   // 다음 항목 이동
else if(sw_in == 4'b0100 && location > 3'd0) 
    location <= location - 1;   // 이전 항목 이동
else if(sw_in == 4'b0010) begin
    if(location == 3'd0 && year_set < 14'd9999)
        year_set <= year_set + 1;   // 증가
    ...
end
else if(sw_in == 4'b0001) begin
    if(location == 3'd0 && year_set > 1)
        year_set <= year_set - 1;   // 감소
    ...
end
```

```Verilog
else if(location == 3'd6) begin
    en_time <= 1'b1;   // 저장 시그널
end
...
else if(location == 3'd6) begin
    year_set  <= year;    // 현재 시간으로 되돌리기
    month_set <= month;
    ...
end

transfer_time <= {year_set, month_set, day_set, hour_set, min_set, sec_set};
```

### **테스트 벤치**
시간 증가 동작을 빠르게 확인하기 위해 사용하였습니다. 시뮬레이션 환경을 `timescale 1ns/10ps`로 설정하고, 파라미터를 `STEP=20ns`를 기준으로 시스템 클록과 1Hz 신호를 생성하였습니다. 리셋 직후 연·월·일·시·분·초 값이 정상적으로 초기화되는 것을 확인할 수 있었으며, 이후 1Hz펄스가 발생할 때마다 초(sec)가 1씩 증가하였습니다. 파형을 통해 초 증가와 초기값 설정 동작이 정상적으로 수행됨을 검증하였습니다. 

```Verilog
`timescale	1ns/10ps

module	tb_watch_time;


	parameter	STEP	= 20;	// 20ns
	
	reg	clk, rst, en_1hz; 
	wire	[13:0]	year; 
	wire	[7:0]	month, day, hour, min, sec, week;



	watch_time	TIME	(
					.clk				(clk),
					.clk1sec				(en_1hz),
					.rst				(rst),
					.year				(year),
					.month				(month),
					.day				(day),
					.hour				(hour),
					.min				(min),
					.sec				(sec),
					.week				(week),
					.max_date			(max_d));

	initial begin
		clk = 0; rst = 1; #(STEP/2);
		rst = 0; #30;
		rst = 1; en_1hz = 1; #(STEP*300000000);
		$stop;
	end

	always		#(STEP)		clk=~clk;
	
	always		#(STEP*50000000)		en_1hz = ~en_1hz;

endmodule
```

## **산출물** 
### **디지털 시계**
![디지털 시계](https://github.com/tae2on/Verilog_clock_system/blob/main/img/mode_watch.png?raw=true)

### **날짜·요일·시간 설정**
![날짜·요일·시간 설정](https://github.com/tae2on/Verilog_clock_system/blob/main/img/mode_watch_set.png?raw=true)

### **알람이 울리면 UART를 통해 메시지를 송신**
![알람이 울리면 UART를 통해 메시지를 송신](https://github.com/tae2on/Verilog_clock_system/raw/main/img/mode_uart_tx_alarm.png?raw=true)

### **UART 수신을 통해 모드를 변경**
![UART 수신을 통해 모드를 변경](https://github.com/tae2on/Verilog_clock_system/blob/main/img/mode_uart.png?raw=true)

### **테스트 벤치**
![테스트 벤치](https://github.com/tae2on/Verilog_clock_system/blob/main/img/test%20bench.png?raw=true)

## **실험 결과**
FPGA 보드에 구현한 디지털 시계는 LCD를 통해 연·월·일·요일·시·분·초가 정상적으로 표시되었으며, 1Hz 클록 신호에 따라 초가 증가하는 동작을 확인하였습니다. 날짜 계산에서도 30일/31일, 윤년의 2월 29일 등 월별 경계 조건을 정확하게 처리하였습니다. 또한 개인 기능인 날짜·요일·시간 설정 모드를 통해 사용자가 원하는 시각을 직접 지정할 수 있었으며, 저장 이후 실제 시계 동작에 반영되는 것도 확인되었습니다. 알람 발생 시 UART를 통해 PC 터미널로 메시지가 전송되고, 외부에서 UART 명령어를 입력해 모드를 전환하는 기능 역시 정상적으로 동작하였습니다. 파형 시뮬레이션과 실험을 종합적으로 통해, 전체 설계가 의도한 대로 안정적으로 수행됨을 검증할 수 있었습니다.

## **고찰**
이번 프로젝트을 통해 검증 과정의 중요성을 깊이 깨달을 수 있었습니다. 첫 프로젝트였던 만큼 오류를 발견하고 해결하는데 많은 시간이 소요하여 진행하는데 많은 어려움이 있었습니다. 이 경험을 통해 사전에 충분한 검증 과정을 거치는 것이 얼마나 중요한지 몸소 느낄 수 있었습니다. 특히 테스트벤치를 활용하여 시간 증가, 초기화, 날짜 경계 조건을 미리 점검함으로써 보드에서의 시행착오를 줄일 수 있었고, 이러한 검증 절차가 프로젝트의 완성도에 큰 영향을 준다는 점을 배울 수 있었습니다. 