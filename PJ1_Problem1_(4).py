## Project1 Problem1 (4) (미래자동차공학과 2019048440 최윤석)
import numpy as np                      # 수학 연산을 위한 numpy 패키지 불러오기
import matplotlib.pyplot as plt         # 시각화를 위한 matplotlib 패키지 불러오기
from scipy.linalg import inv            # 역행렬 계산을 위한 scipy의 linalg 모듈에서 inv 함수 불러오기

# 첫 번째 함수 정의: 이차함수 형태
def function1(x):
    x1, x2 = x                          # 입력 벡터를 x1, x2로 분리
    return 1 + 2*x1 + 3*(x1**2 + x2**2) + 4*x1*x2  # 함수식 계산 후 반환

# 첫 번째 함수의 기울기(Gradient) 정의
def gradient1(x):
    x1, x2 = x
    df_dx1 = 2 + 6*x1 + 4*x2            # x1에 대한 편미분 결과
    df_dx2 = 6*x2 + 4*x1                # x2에 대한 편미분 결과
    return np.array([df_dx1, df_dx2])   # gradient 벡터 반환

# 두 번째 함수 정의: 행렬/벡터 기반 이차함수
def function2(x):
    Q = np.array([[3, 3], [1, 3]])      # 2x2 대칭 행렬 Q 정의
    b = np.array([16, 23])              # 선형 항 벡터 b 정의
    return x.T @ Q @ x + b.T @ x + np.pi**2  # 이차항 + 선형항 + 상수항

# 두 번째 함수의 gradient 정의
def gradient2(x):
    Q = np.array([[3, 3], [1, 3]])      # 같은 Q 행렬
    b = np.array([16, 23])              # 같은 b 벡터
    return 2 * Q @ x + b                # 경사 벡터 계산 공식

# 세 번째 함수 정의: 명시적 이차식
def function3(x):
    x1, x2 = x
    return 3*(x1**2 + x2**2) + 4*x1*x2 + 5*x1 + 6*x2 + 7

# 세 번째 함수의 gradient 정의
def gradient3(x):
    x1, x2 = x
    df_dx1 = 6*x1 + 4*x2 + 5            # x1에 대한 편미분 결과
    df_dx2 = 6*x2 + 4*x1 + 6            # x2에 대한 편미분 결과
    return np.array([df_dx1, df_dx2])   # gradient 벡터 반환

# 경사 하강법(Gradient Descent) 알고리즘 정의
def gradient_descent(func, grad, x0, alpha=0.1, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)       # 시작점 x0를 float 타입 배열로 변환
    values = []                         # 함수값 저장용 리스트 초기화
    for _ in range(max_iter):           # 최대 반복 횟수만큼 루프 실행
        values.append(func(x))          # 현재 함수값 저장
        grad_x = grad(x)                # 현재 지점에서의 gradient 계산
        if np.linalg.norm(grad_x) < tol:  # gradient 크기가 작아지면 수렴했다고 판단
            break                       # 수렴 시 반복 중단
        x -= alpha * grad_x             # 경사 하강법 업데이트 (x = x - α * grad)
    return x, values                    # 최적해와 함수값 리스트 반환

# 세 개의 함수 및 gradient 리스트 구성
functions = [(function1, gradient1), (function2, gradient2), (function3, gradient3)]
x0_list = [[0, 0], [0, 0], [0, 0]]      # 각 함수에 대해 초기값 x0 설정
names = ["Function 1", "Function 2", "Function 3"]  # 그래프 및 출력용 이름 설정

plt.figure(figsize=(12, 5))            # 전체 그래프 크기 설정

# 각 함수마다 경사 하강법 실행 및 결과 출력
for i, (func, grad) in enumerate(functions):
    x0 = x0_list[i]                    # 해당 함수의 초기값
    gd_x, gd_vals = gradient_descent(func, grad, x0)  # 경사 하강법 실행

    # 최적해와 최적 함수값 출력
    optimal_value = func(gd_x)
    print(f"{names[i]}:")
    print(f"  Optimal Solution: x* = {gd_x}")
    print(f"  Optimal Value: f(x*) = {optimal_value}\n")
    
    # 함수값 변화 그래프 서브플롯으로 그리기
    plt.subplot(1, 3, i+1)
    plt.plot(gd_vals, label='Gradient Descent', linestyle='dashed')  # 함수값 그래프
    plt.xlabel('Iterations')           # x축 라벨
    plt.ylabel('Function Value')       # y축 라벨
    plt.title(names[i])                # 그래프 제목
    plt.legend()                       # 범례 표시

plt.tight_layout()                     # 레이아웃 자동 정렬
plt.show()                             # 그래프 출력