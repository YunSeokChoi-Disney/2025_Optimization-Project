###Project 1-Problem3 (미래자동차공학과 2019048440 최윤석)

import numpy as np     # Imports the numpy library for numerical calculation as 'np'
import matplotlib.pyplot as plt   # Imports the plotting module from matplotlib and aliases it as 'plt'

#x1 as x, x2 as y
def f(x,y): #define function for problem 3
    return 100*(y-x**2)**2+(1-x)**2 #return result

def gradient(x,y): #define gradient of function
    grad_x = -400*(y-x**2)*x-2*(1-x) #partial derivate of x
    grad_y = 200*(y-x**2) #partial derivate of y
    return (grad_x,grad_y) #return gradient

def gradient_descent(initial_point, learning_rate, max_iteration, target_error): #gradient descent method algorithm
    x_value, y_value = [], [] #set x, y value array
    iteration=0 #set number of iteration as 0
    error=float('inf') #set error as infinity
    x, y = initial_point #set initial point
    x_value.append(x) #append x_value array
    y_value.append(y) #append y_value array
    for i in range(max_iteration): #run gradient descent iteration
        grad_x, grad_y = gradient(x,y) #calculate gradient
        x=x-learning_rate*grad_x #calculate new x
        y=y-learning_rate*grad_y #calculate new y
        x_value.append(x) #append x_value array
        y_value.append(y) #append y_value array
        error=abs(f(x,y)-f(x_value[i],y_value[i])) #calculate error
        iteration=i #renew number of iteration
        if error<=target_error: #break loop condition
            break #stop the loop
    return x_value, y_value, iteration #return x_value, y_value, number of iteration

#set arbitrary initial point
xi=1.5 #initial x point
yi=3.5 #initial y point
initial_point = xi, yi #set initial point
learning_rate = 0.001 #set learning rate
max_iteration = 1000000 #set max number of interation
target_error = 1e-6 #set target error

x_value, y_value, iteration = gradient_descent(initial_point, learning_rate, max_iteration, target_error) #run gradient descent method algorithm

print('iteration =',iteration) #print number of interation
print('initial_point(x1,x2) = (', xi, ',', yi,')') #print initial point
print('opt_point(x1,x2) = (', x_value[len(x_value)-1],',', y_value[len(y_value)-1],')') #print optimization point
print('opt_value =', f(x_value[len(x_value)-1], y_value[len(y_value)-1])) #print optimization value

plt.figure(figsize=(10,6)) #set plot size
plt.scatter(xi, yi, c='r', label='Initial Point') #plot initial point
plt.scatter(x_value[len(x_value)-1], y_value[len(y_value)-1], c='g', label='Optimization Point') #plot optimization point
plt.plot(x_value, y_value, c='b', label='Optimization Path') #plot optimization path
plt.title('Gradient Descent Optimization') #set title
plt.xlabel('x1') #set x label
plt.ylabel('x2') #set y label
plt.legend() #plot legend
plt.show() #show plot