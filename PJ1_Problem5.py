##Project1 Problem 5 (미래자동차공학과 2019048440 최윤석)

##### Project 1, Problem 5-3, First Method (Analytic Solution) ######

import numpy as np #numpy 불러오기

# Setting the dimensions
n, m = 100, 50 #차원 설정

# Generate Gaussian random data for A and b
A = np.random.randn(m, n) #A 랜덤행렬 생성
b = np.random.randn(m, 1) #b 랜덤벡터 생성

# Generate a symmetric positive definite matrix Q
random_matrix = np.random.randn(m, m) #랜덤Q 생성
Q = np.dot(random_matrix, random_matrix.T)  # Q is symmetric and positive definite

# Compute the analytical solution x_analytical = (A^T Q A)^-1 A^T Q b
A_T_Q = A.T @ Q #A전치와 Q 곱
A_T_Q_A = A_T_Q @ A #A전치Q와 A 곱
A_T_Q_b = A_T_Q @ b #A전치Q와 b 곱
x_analytical = np.linalg.inv(A_T_Q_A) @ A_T_Q_b # (A^T Q A)^-1 A^T Q b

A_t=np.transpose(A) #A전치
Anew=A_t@A #A전치A 생성
Ad=np.linalg.det(Anew) #A전치A의 행렬식

print(x_analytical)  # Display the analytical solution
print(Ad) #A전치A의 행렬식 출력

###### Project 1, Problem 5-3, Second Method (Gradient Descent) #######

import numpy as np

def gradient_descent(A, b, Q, x_init, learning_rate, max_iter, tol): #경사하강법 시작
    x = x_init #초깃값 선언
    f_prev = float('inf') #'무한'을 나타내는 값 만들어주기
    
    for i in range(max_iter): #최대 반복횟수까지 연산
        # Compute the gradient
        grad = 2 * A.T @ Q @ (A @ x - b) #그라디언트 선언
        
        # Update x
        x = x - learning_rate * grad # x 업데이트
        
        # Compute the value of the function
        f_x = ((A @ x - b).T @ Q @ (A @ x - b)).item() #목적함수 선언
        
        # Check for convergence
        if abs(f_prev - f_x) < tol: #만약 오차허용범위보다 값들의 차이가 작으면
            break #계산 중지
        f_prev = f_x #최종값 반환
        
       
    
    return x, i+1  # Return the solution and the number of iterations

# Initial guess for x
x_init = np.random.randn(n, 1) #x0 초깃값 선언

# Parameters for gradient descent
learning_rate = 0.001 #학습률은 0.001로 설정
max_iter = 1000 # 최대반복횟수는 1000회
tol = 1e-6 #오차 허용 범위는 1e-6

# Perform gradient descent
x_gd, iterations = gradient_descent(A, b, Q, x_init, learning_rate, max_iter, tol) #결과 계산함수

x_gd, iterations  # Display the gradient descent solution and number of iterations

A_t=np.transpose(A) # A 전치 생성
Anew=A_t@A # A전치A 행렬 만들기
Ad=np.linalg.det(Anew) # 행렬식 계산

print(x_gd, iterations)  # Display the analytical solution
print(Ad) # 행렬식 선언


###### Project 1, Problem 5-4 #######

import numpy as np # numpy 라이브러리 호출

# Redefine dimensions for the new problem setup
n, m = 50, 100 #차원 설정 (3번)

# Generate Gaussian random data for A and b with new dimensions
A = np.random.randn(m, n) #랜덤 A 생성
b = np.random.randn(m, 1) #랜덤 B 생성 

# Generate a symmetric positive definite matrix Q (m x m)
random_matrix = np.random.randn(m, m) # 랜덤 Q 생성 
Q = np.dot(random_matrix, random_matrix.T)  # Q remains symmetric and positive definite

# Compute the analytical solution x_analytical = (A^T Q A)^-1 A^T Q b
A_T_Q = A.T @ Q # A^T Q 곱 계산
A_T_Q_A = A_T_Q @ A # A^TQA 계산
A_T_Q_b = A_T_Q @ b # A^TQb 계산
x_analytical_new = np.linalg.inv(A_T_Q_A) @ A_T_Q_b #해석해 출력

# Define gradient descent parameters
learning_rate = 1e-5 #학습률
max_iter = 1000 #최대 반복횟수
tol = 1e-6 #오차범위
x_init_new = np.random.randn(n, 1)  # Reset initial guess for x

# Perform gradient descent with the new setup
def gradient_descent_very_safe(A, b, Q, x_init, learning_rate, max_iter, tol): #경사하강법을 실시하는 함수 정의
    x = x_init #초깃값 생성
    f_prev = float('inf') #무한대 구현
    
    for i in range(max_iter): #최대 반복 횟수까지 반복
        # Compute the gradient
        grad = 2 * A.T @ Q @ (A @ x - b) #그라디언드 선언
        
        # Check for extreme gradient values and adjust learning rate if necessary
        if np.linalg.norm(grad) > 1e10:  # Threshold for considering gradient as exploded
            learning_rate *= 0.1  # Reduce learning rate
            print(f"Reducing learning rate to: {learning_rate}") # 학습률이 줄어든 경우 줄어든 값을 출력
        
        # Update x
        x = x - learning_rate * grad # 경사하강법 실시
        
        # Compute the value of the function
        f_x = ((A @ x - b).T @ Q @ (A @ x - b)).item() #목적함수 값 계산
        
        # Check for convergence
        if abs(f_prev - f_x) < tol: #만일 오차가 tolerance보다 작다면
            break #작동 중지
        f_prev = f_x #값 반환
    
    return x, i+1  # Return the solution and number of iterations

x_gd_new, iterations_new = gradient_descent_very_safe(A, b, Q, x_init_new, learning_rate, max_iter, tol) #결과 계산 

# Error Calculation
C=x_analytical_new-x_gd_new # 해석적 방법과 경사하강법의 차이 계산
C_t=np.transpose(C) # 행렬의 오차 계산, 즉 내적을 위한 전치 행렬 생성
Error=abs((C_t.dot(C))**2) # 오차 계산

A_t=np.transpose(A) #A전치
Anew=A_t@A #A^TA 계산
Ad=np.linalg.det(Anew) #det(A^T A) 계산 

# Display the new analytical solution, GD solution, iterations, and Error
print(x_analytical_new) # 해석적 방법 출력
print(x_gd_new) # 경사하강법 해 출력
print(iterations_new) # 반복 횟수 출력
print(Error)# 오차 출력
print(Ad)# A의 행렬식 출력