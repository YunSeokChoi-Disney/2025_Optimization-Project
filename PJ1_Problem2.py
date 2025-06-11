##Project1 Problem2 (미래자동차공학과 2019048440 최윤석)

import numpy as np      # 수학 연산을 위한 numpy 패키지 불러오기
import cvxpy as cp      # cvxpy 패키지 불러오기 for convex optimization
from math import sqrt    #math 모듈에서 sqrt(제곱근 함수)만 불러오기 
import matplotlib.pyplot as plt  #matplotlib의 pyplot 모듈 불러오기 

A = np.random.randn(1000,1000) #random 1000*1000 matrix
while np.linalg.matrix_rank(A) != 1000: #recreate A if not full rank until A is full rank
    A = np.random.randn(1000,1000) #recreate A if not full rank

rho = np.random.uniform(low = 0.0, high = 1000.0) #random positive scalar for Q to be positive definite

Q = np.dot(A, A.T) + rho * np.identity(1000) #create matrix Q
b = np.random.randn(1000) #random 1000 dim vector b
initial_x = np.random.randn(1000) #random initial input for x

def function2(x: np.ndarray): #function to calc function result
    result = 0.5 * np.dot(x.T, np.dot(Q,x)) - np.dot(b.T, x)
    gradient = np.dot(Q,x) - b
    return result, gradient

#obtain optimal solution utilizing CVXPY library
# https://www.cvxpy.org/examples/
x_cvx = cp.Variable(1000) #delcare x_cvx as variable
function_cvx = 0.5 * ((x_cvx.T) @ Q @ x_cvx) - (b.T @ x_cvx) #declare problem
prob = cp.Problem(cp.Minimize(function_cvx)) #define problem
prob.solve() #solve
print("CVXPY Result:\n", "Opt Value: ", prob.value) #print cost
#print("\nX: ",x_cvx.value)

#gradient descent algorithm
max_iteration = 500 #Number of iteration
stepsize = 0.0001 #step size value
epsilon = 0.001 #epsilon value for stopping criteria

#initialize x,y matrices
x_gradient = np.zeros(shape = (1000, max_iteration), dtype = np.double) #initialize x
y_gradient = np.zeros(shape = (max_iteration,), dtype = np.double) #initialize y
x_gradient[:,0] = initial_x
y_gradient[0], gradient = function2(x_gradient[:,0]) #initial y

for n in range(1, max_iteration):       #run gradient descent algorithm
    x_gradient[:,n] = x_gradient[:,n-1] - stepsize * gradient   #calculate new x vextor
    y_gradient[n], gradient = function2(x_gradient[:,n])    #calculate new y
    if abs(y_gradient[n]-y_gradient[n-1]) < epsilon:    #stopping criterion
        iteration_end_GD = n+1             #note when iteration stopped
        break                            #stop the loop
else:
    iteration_end_GD = max_iteration    #if the convergence condition is not satisfied

x_gradient = x_gradient[:, :iteration_end_GD]   #cut x according to # of iteration
y_gradient = y_gradient[:iteration_end_GD]      #cut y according to # of iteration


#show result
print("GP- Number of Iteration: ", iteration_end_GD) #print how much iteration it took
print("result: ", y_gradient[iteration_end_GD-1])
delta_gradient = np.mean(np.absolute(x_gradient - np.repeat(np.array(x_cvx.value).reshape(1000,1), iteration_end_GD, axis =1)),axis = 0) #obtain abs(x_k -x*)
#show convergence plot

iteration_gradient = range(0,iteration_end_GD) #array for X-axis plot
plt.subplot(2,2,1)#plot top left
plt.plot(iteration_gradient, y_gradient, 'r',label = 'function')#convergence plot of function
plt.plot(iteration_gradient, delta_gradient, 'b', label = 'mean(delta x)')#convergence plot of abs(x-x*)
plt.legend()#plot legend

#steepest GD
#initialize x,y matrices
x_sgd = np.zeros(shape = (1000, max_iteration), dtype = np.double) #initialize x
y_sgd = np.zeros(shape = (max_iteration,), dtype = np.double) #initialize y
x_sgd[:,0] = initial_x
y_sgd[0], gradient = function2(x_sgd[:,0]) #initial y

for n in range(1, max_iteration):      #run gradient_descent alogorithm
    stepsize_sgd = np.dot(gradient.T, gradient)/np.dot(gradient.T,np.dot(Q,gradient)) #calc steepest GD stepsize
    x_sgd[:,n] = x_sgd[:,n-1] - stepsize_sgd * gradient     #calculate new x vector
    y_sgd[n], gradient = function2(x_sgd[:,n])             #calculate new y    
    if abs(y_sgd[n]-y_sgd[n-1]) < epsilon:     #stopping criterion
        iteration_end_sgd = n+1               #note when iteration stopped
        break                                 #stop the loop
else:
    iteration_end_sgd = max_iteration       #if the convergence condition is not satisfied

x_sgd = x_sgd[:, :iteration_end_sgd]        #cut x according to # of iteration
y_sgd = y_sgd[:iteration_end_sgd]           #cut y according to # of iteration


#show result
print("Steepest GD- Number of Iteration: ", iteration_end_sgd) #print how much iteration it took
print("result: ", y_sgd[iteration_end_sgd-1])

delta_sgd = np.mean(np.absolute(x_sgd - np.repeat(np.array(x_cvx.value).reshape(1000,1), iteration_end_sgd, axis =1)),axis = 0) #obtain abs(x_k -x*)
#show convergence plot
iteration_sgd = range(0,iteration_end_sgd) #array for X-axis plot
plt.subplot(2,2,2) #plot top right
plt.plot(iteration_sgd, y_sgd, 'r',label = 'function')#convergence plot of function
plt.plot(iteration_sgd, delta_sgd, 'b', label = 'mean(delta x)')#convergence plot of abs(x-x*)
plt.legend()#plot legend

#Nesterov-2 Algorithm
stepsize = 0.0005 #step size
alpha = np.array([0.5, 0]) #initial alpha value = 0.5 #initial beta value

#initialize x,y matrices
x_nesterov = np.zeros(shape = (1000, max_iteration), dtype = np.double) #initialize x matrix
f = np.zeros(shape = (max_iteration,), dtype = np.double) #vector for tracking function result
y = np.zeros(shape = (1000,2), dtype = np.double) #1000*2 for y_k and y_(k+1)
x_nesterov[:,0] = np.random.randn(1000) #randomize initial input for x
f[0], gradient = function2(x_nesterov[:,0]) #initial function value

#run Nesterov-2 algorithm
for n in range(1, max_iteration):    
    k = (n-1)%2    #calculation for alternating value of k
    k2 = n%2        #calculation for alternating value of k+1(for saving memory)
    stepsize = np.dot(gradient.T, gradient)/np.dot(gradient.T,np.dot(Q,gradient))   #calculate steepest step size
    alpha[k2] = 0.5*(sqrt(alpha[k]**4 + 4*(alpha[k]**2))-alpha[k]**2)    #calculate alpha_k+1
    beta = alpha[k]*(1-alpha[k])/(alpha[k]**2 + alpha[k2])               #calculate beta_k
    y[:,k2] = x_nesterov[:,n-1] - stepsize * gradient                    #calculate y_(k+1)
    x_nesterov[:,n] = y[:,k2] + beta * (y[:,k2]-y[:,k])                  #calculate x_(k+1)
    f[n], gradient = function2(x_nesterov[:,n])                          #calculate function cost and gradient
    if abs(f[n]-f[n-1]) < epsilon:                                       #stopping criterion
        iteration_end_N2 = n+1                                           #note when iteration stopped
        break                                                            #stop the loop
else:
    iteration_end_N2 = max_iteration                                      #if the convergence condition is not satisfied

x_nesterov = x_nesterov[:,:iteration_end_N2] #discard unused memory
f = f[:iteration_end_N2] #discard unused memory
#show result
print("Nesterov-2 Algorithm -Number of Iteration: ", iteration_end_N2) #print number of iteration
print("result: ", f[iteration_end_N2 -1]) #print result cost

delta_nesterov = np.mean(np.absolute(x_nesterov - np.repeat(np.array(x_cvx.value).reshape(1000,1), iteration_end_N2, axis =1)),axis = 0) #obtain abs(x_k -x*)
#show convergence plot
iteration_N2 = range(0,iteration_end_N2) #array for X-axis plot
plt.subplot(2,2,3) #plot bottom left
plt.plot(iteration_N2, f, 'r',label = 'function')#convergence plot of function
plt.plot(iteration_N2, delta_nesterov, 'b', label = 'mean(delta x)')#convergence plot of abs(x-x*)
plt.legend() #plot legend

#compare results between algorithms
plt.subplot(2,2,4) #plot bottom
plt.plot(iteration_gradient, delta_gradient,'r', label = 'GD') #plot mean error of gradient descent
plt.plot(iteration_sgd, delta_sgd, 'g', label = 'Steepest GD') #plot mean error of steepest gradient descent
plt.plot(iteration_N2, delta_nesterov, 'b', label = 'Nesterov-2') #plot mean error of steepest gradient descent

plt.legend() #plot legend
plt.show() #show plot