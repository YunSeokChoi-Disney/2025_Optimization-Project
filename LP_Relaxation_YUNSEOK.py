###Project2 LP for Knapsack Problem(2019048440 CHOI YUN SEOK)###
 
#Import modules
import numpy as np  # Importing NumPy for array operations
import time  # Importing the time module for time measurement
from scipy.optimize import linprog  # Importing the linprog function for linear programming

#Set LP Relaxation Function 
def knapsack_lp_relaxation(weights, values, capacity):
    n = len(weights)  # Number of items

    # Objective function: -1 * (value[0]*x1 + value[1]*x2 + ... + value[n]*xn)
    # Objective function: Array with negative values representing the values of each item
    c = -1 * values

    # Inequality constraints: weights[0]*x1 + weights[1]*x2 + ... + weights[n]*xn <= capacity
    # Inequality constraints: Matrix A and vector b representing the relationship between item weights and knapsack capacity
    A = weights.reshape(1, -1)
    b = np.array([capacity])

    # Bounds: 0 <= xi <= 1
    # Bounds: Lower and upper bounds for each variable xi
    bounds = [(0, 1)] * n

    # Solving the LP relaxation problem using linear programming
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # The optimal knapsack cost is stored as a negative value in the result of the LP relaxation problem
    max_value = -result.fun
    Solution = [item+1 for item in range(n) if result.x[item]>0] # Indices of items included in the solution
    return max_value, Solution

if __name__ == "__main__":
    np.random.seed(42)
    size = 7 # Number of items
    weights = np.random.randint(1, 50, size=size)  # Randomly generated weights
    values = np.random.randint(1, 100, size=size)  # Randomly generated values
    capacity = 9  #Maximum Capacity of Knapsack

    start_time = time.time()  # Recording the start time of the algorithm
    lp_relaxation_cost, lp_relaxation_solution = knapsack_lp_relaxation(weights, values, capacity)  
    # Solving the Knapsack problem using LP relaxation
    lp_relaxation_time = time.time() - start_time  # Calculating the execution time of the algorithm

    print("LP Relaxation Cost:", lp_relaxation_cost)  # Printing the optimal Knapsack cost using LP relaxation
    print("LP Relaxation Solution:", lp_relaxation_solution) # Printing the optimal Knapsack solution using LP relaxation
    print("LP Relaxation Time:", lp_relaxation_time)  # Printing the execution time of the LP relaxation algorithm