##Project1 Problem1 (5) (미래자동차공학과 2019048440 최윤석)
import numpy as np                      # 수치 계산을 위한 numpy 불러오기
import matplotlib.pyplot as plt         # 그래프 시각화를 위한 matplotlib 불러오기
from scipy.linalg import inv            # 행렬의 역행렬 계산을 위한 함수 불러오기

# 함수 1 정의 및 그라디언트 
def function1(x):
    x1, x2 = x
    return 1 + 2*x1 + 3*(x1**2 + x2**2) + 4*x1*x2  # 이차 다항식 형태의 함수

def gradient1(x):
    x1, x2 = x
    return np.array([6*x1 + 4*x2 + 2, 6*x2 + 4*x1])  # 각 변수에 대한 편미분 결과 (gradient 벡터)

# 함수 2 정의 및 그라디언트 
def function2(x):
    Q = np.array([[3, 3], [1, 3]])  # 이차항 계수 행렬 Q
    b = np.array([16, 23])          # 선형항 계수 벡터 b
    return x.T @ Q @ x + b.T @ x + np.pi**2  # 이차항 + 선형항 + 상수

def gradient2(x):
    Q = np.array([[3, 3], [1, 3]])
    b = np.array([16, 23])
    return 2 * Q @ x + b            # 함수의 gradient 공식

# 함수 3 정의 및 그라디언트 
def function3(x):
    x1, x2 = x
    return 3*(x1**2 + x2**2) + 4*x1*x2 + 5*x1 + 6*x2 + 7  # 이차 다항식

def gradient3(x):
    x1, x2 = x
    return np.array([6*x1 + 4*x2 + 5, 6*x2 + 4*x1 + 6])  # 편미분해서 gradient 벡터 반환

#  각 함수의 Hessian 행렬 (2차 미분 행렬) 
def hessian1():
    return np.array([[6, 4], [4, 6]])  # 함수 1의 Hessian (상수 행렬)

def hessian2():
    return 2 * np.array([[3, 3], [1, 3]])  # 함수 2의 Hessian (Q의 2배)

def hessian3():
    return np.array([[6, 4], [4, 6]])  # 함수 3의 Hessian (상수 행렬)

# Steepest Descent 알고리즘 정의 
def steepest_descent(func, grad, hessian, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)        # 초기값 x0를 float 형으로 설정
    values = []                          # 함수값 저장 리스트

    for _ in range(max_iter):            # 최대 반복 횟수만큼 반복
        values.append(func(x))           # 현재 함수값 저장
        grad_x = grad(x)                 # 현재 위치에서의 gradient 계산

        if np.linalg.norm(grad_x) < tol: # gradient 벡터의 크기가 작으면 수렴한 것으로 판단
            break                        # 반복 종료

        H_inv = inv(hessian())           # Hessian 행렬의 역행렬 계산
        # Steepest descent에서 optimal step size 계산 공식 (quadratic function에만 사용 가능)
        alpha = (grad_x.T @ H_inv @ grad_x) / (grad_x.T @ H_inv @ H_inv @ grad_x)

        x -= alpha * grad_x              # 스텝 사이즈 * gradient 방향으로 이동
    return x, values                     # 최적해와 함수값 리스트 반환

#  함수/gradient/hessian을 리스트로 묶기 
functions = [
    (function1, gradient1, hessian1),
    (function2, gradient2, hessian2),
    (function3, gradient3, hessian3)
]

x0_list = [[0, 0], [0, 0], [0, 0]]       # 각 함수에 대한 초기값 설정
names = ["Function 1", "Function 2", "Function 3"]  # 함수 이름 리스트

plt.figure(figsize=(12, 5))              # 그래프 크기 설정

# 각 함수에 대해 Steepest Descent 수행 및 결과 시각화 
for i, (func, grad, hessian) in enumerate(functions):
    x0 = x0_list[i]                      # 해당 함수의 초기값
    sd_x, sd_vals = steepest_descent(func, grad, hessian, x0)  # 알고리즘 실행

    # 최적해 및 최적 함수값 출력
    optimal_value = func(sd_x)
    print(f"{names[i]}:")
    print(f"  Optimal Solution: x* = {sd_x}")
    print(f"  Optimal Value: f(x*) = {optimal_value}\n")
     
    # 그래프 subplot에 결과 그리기
    plt.subplot(1, 3, i+1)
    plt.plot(sd_vals, label='Steepest Descent', linestyle='solid')  # 함수값 변화 그래프
    plt.xlabel('Iterations')            # x축 라벨
    plt.ylabel('Function Value')        # y축 라벨
    plt.title(names[i])                 # 그래프 제목
    plt.legend()                        # 범례 표시

plt.tight_layout()                      # subplot 사이 여백 정리
plt.show()                              # 최종 그래프 출력
