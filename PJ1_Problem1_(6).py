##Project1 Problem1 (6) (미래자동차공학과 2019048440 최윤석)
import numpy as np                          #  수치 계산을 위한 numpy 불러오기
import matplotlib.pyplot as plt             # 그래프 시각화를 위한 matplotlib 불러오기

#  함수 1 정의 및 Gradient 
def func1(x1, x2):
    return (x1 + 0.6)**2 + (x2 - 0.4)**2     # 최소값은 (-0.6, 0.4)

def grad_func1(x1, x2):
    return np.array([2 * (x1 + 0.6), 2 * (x2 - 0.4)])  # 각 변수에 대한 편미분

# === 함수 2 정의 및 Gradient ===
def func2(x1, x2):
    return (x1 + 0.2) + (x2 + 3.7)**2        # 선형 + 이차항

def grad_func2(x1, x2):
    return np.array([2 * (x1 + 0.2), 2 * (x2 + 3.7)])  # gradient 계산

#  함수 3 정의 및 Gradient 
def func3(x, y):
    return (x + 0.3)**2 + (y + 0.8)**2       # 최소값은 (-0.3, -0.8)

def grad_func3(x, y):
    return np.array([2 * (x + 0.3), 2 * (y + 0.8)])  # gradient 계산

#  Steepest Gradient 방향 벡터 계산 
def steepest_gradient(grad):
    norm = np.linalg.norm(grad)             # gradient의 크기 계산
    return -grad / norm if norm != 0 else np.zeros_like(grad)  # 크기로 나눠 단위 벡터로 변환 (steepest descent)

# 각 함수의 최적해 지정 
opt_solutions = {
    'func1': np.array([-0.6, 0.4]),
    'func2': np.array([-0.2 -3.7]),      
    'func3': np.array([-0.3, -0.8])
}

# 반복 횟수 및 학습률 설정
num_iterations = 10000                        # Gradient Descent 반복 횟수 1000회로 설정정
learning_rate = 0.1                         # 학습률 0.1로 설정정

# plot 생성 (2행 3열 서브플롯)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 각 함수에 대해 Gradient 분석
for i, (name, opt_x) in enumerate(opt_solutions.items()):
    x_vals, grad_vals, steepest_vals = [], [], []              # 거리 저장 리스트 초기화
    x = opt_x + np.array([1.0, -1.0])                          # 최적해 근처의 초기값 설정

    for _ in range(num_iterations):
        grad = globals()[f'grad_{name}'](*x)                   # gradient 함수 호출
        steepest_grad = steepest_gradient(grad)               # steepest gradient 계산

        x_vals.append(np.linalg.norm(x - opt_x))              # 현재 위치와 최적해 사이 거리 저장
        grad_vals.append(np.linalg.norm(x - grad))            # 현재 위치와 grad 방향 차이 저장
        steepest_vals.append(np.linalg.norm(x - steepest_grad))  # 현재 위치와 steepest 방향 차이 저장

        x -= learning_rate * grad                              # gradient descent update

    # 최적해 및 함수값 계산
    optimal_solution = x
    optimal_value = globals()[name](*optimal_solution)

    # 결과 출력
    print(f"{name}:")
    print(f"  Optimal Solution: x* = {optimal_solution}")
    print(f"  Optimal Value: f(x*) = {optimal_value}\n")

    #  [1행] Gradient 방향 vs Steepest 방향 차이 플롯 
    axes[0, i].plot(range(num_iterations), np.abs(np.array(x_vals) - np.array(grad_vals)), label='|opt_x - grad|')
    axes[0, i].plot(range(num_iterations), np.abs(np.array(x_vals) - np.array(steepest_vals)), label='|opt_x - steepest_grad|', linestyle='dashed')
    axes[0, i].set_title(f'{name} Gradient Difference')        # 그래프 제목
    axes[0, i].legend()
    axes[0, i].set_xlabel('Iterations')
    axes[0, i].set_ylabel('Difference')

    #  [2행] 함수값 수렴 플롯 
    func_vals = [globals()[name](*opt_x + np.array([1.0, -1.0]))]  # 초기 함수값
    x = opt_x + np.array([1.0, -1.0])                              # 초기값 재설정
    for _ in range(num_iterations):
        grad = globals()[f'grad_{name}'](*x)                       # gradient 계산
        x -= learning_rate * grad                                  # 업데이트
        func_vals.append(globals()[name](*x))                      # 함수값 저장

    axes[1, i].plot(range(num_iterations + 1), func_vals, label='Function Value')  # 함수 수렴 그래프
    axes[1, i].set_title(f'{name} Convergence Plot')    #그래프 제목 'Convergence Plot' 설정정
    axes[1, i].legend()    # 설정된 label을 기반으로 범례 표시
    axes[1, i].set_xlabel('Iterations')   #x축에는 반복횟수 라벨 표시
    axes[1, i].set_ylabel('Function Value')       #x축에는 반복횟수 라벨 표시
 
# 전체 레이아웃 정리 및 출력
plt.tight_layout()
plt.show()
