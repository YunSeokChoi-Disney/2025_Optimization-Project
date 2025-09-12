%%%% Knapsack problem solved using branch and bound algorithm in
%%%% MATLAB(2019048440 CHOI YUN SEOK)

% 문제 데이터
weights = [2, 3, 6, 7, 5, 9, 4]';   % 무게 (열벡터로 변환)
values  = [6, 5, 8, 9, 6, 7, 3]';   % 가치 (열벡터로 변환)
W = 9;                              % 배낭의 최대 무게
num_items = length(weights);       % 아이템 수

% 목적 함수 계수 설정 (최대화를 위해 음수로 변환 → 최소화 문제로 바꿈)
f = -values;

% 제약 조건 설정: A*x ≤ b 형태, 즉 총 무게 ≤ 배낭 용량
A = weights';
b = W;

% 정수형 변수 설정: 0 또는 1 (이진 변수)
intcon = 1:num_items;

% 변수의 하한 및 상한: 0 ≤ x ≤ 1
lb = zeros(num_items, 1);
ub = ones(num_items, 1);

% 정수계획 문제 풀기
[x, fval] = intlinprog(f, intcon, A, b, [], [], lb, ub);

% 선택된 아이템 인덱스 추출
selected_items = find(x > 0.5);

% 결과 출력
disp("선택된 아이템 인덱스:");
disp(selected_items);
disp("총 가치:");
disp(-fval);  % 음수 부호 원래대로 복원
disp("총 무게:");
disp(weights' * x);


