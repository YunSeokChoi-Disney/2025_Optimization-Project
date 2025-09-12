###Project2 Knapsack Problem by using Greedy Algorithm (2019048440 CHOI YUN SEOK)###

import numpy as np  # Import NumPy to create arrays
import time  # Import module for time measurement

# Sort w and p
def Sort(P:np.ndarray, W:np.ndarray):  # Function to sort weights and values
    arr = np.vstack((P,W, P/W))  # Create one array
    arr = Quicksort(arr)  # Perform QuickSort
    return arr[0,:].astype(int), arr[1,:].astype(int)  # Return sorted lists

def Quicksort(array: np.ndarray):  # Implementation of Quicksort function
    if array.shape[1]<=1:  # Check if there's only one item
        return array  # Return if only one item is present
    pivot = array[2,0]  # Set the first column as pivot
    left = np.array([x for x in array[:, 1:].T if len(x) > 0 and x[2] > pivot]).T  # Assign to left array if ratio is greater than pivot
    right = np.array([x for x in array[:, 1:].T if len(x) > 0 and x[2] <= pivot]).T  # Assign to right array if ratio is less than or equal to pivot
    if left.ndim == 1:  # If left array is empty
        return np.hstack((array[:,0].reshape((3,-1)), Quicksort(right)))  # Recursion
    elif right.ndim == 1:  # If right array is empty
        return np.hstack((Quicksort(left), array[:,0].reshape((3,-1))))  # Recursion
    else:  # If both arrays are not empty
        return np.hstack((Quicksort(left), array[:,0].reshape((3,-1)), Quicksort(right)))  # Recursion
    
#Set Greedy Function

def knapsack_greedy(weights, values, capacity):
    n = len(weights)  # Number of items
    ratios = [(values[i] / weights[i], weights[i], values[i],i) for i in range(n)]  # Calculate value to weight ratios of items
    # ratios.sort(reverse=True, key=lambda x: x[0])  
    # Sort in descending order based on ratio -- 
    # This part seems to be commented out as the professor might want the sorting code implemented directly

    max_value = 0  # Initialize maximum value
    solution = []  # Initialize solution matrix to maximize value
    total_weight = 0  # Initialize total weight
    for ratio, weight, value, index in ratios:  # Iterate through each item
        if total_weight + weight <= capacity:  # If current item can be accommodated in the knapsack
            max_value += value  # Accumulate value
            total_weight += weight  # Accumulate weight
            solution.append(index)
        # Greedy algorithm moves to the next item without putting it in the knapsack if it cannot be accommodated
        # else:  # If current item cannot be accommodated in the knapsack
        #     remaining_capacity = capacity - total_weight  # Calculate remaining capacity
        #     max_value += ratio * remaining_capacity  # Accumulate value based on ratio for the remaining capacity
        #     break  # No need to consider further items

    return max_value, solution  # Return maximum value



#Generate random weights and values
def generate_knapsack_data(size,capacity):
    np.random.seed(42)  # Set random seed for reproducibility
    weights = np.random.randint(1, 50, size=size)  # Generate weights randomly
    values = np.random.randint(1, 100, size=size)  # Generate values randomly
    # Regenerate weights if the total weight of items is less than or equal to the capacity of the knapsack
    while np.sum(weights)<=capacity: weights = np.random.randint(1, 50, size=size)  # Generate weights randomly
    return weights, values  # Return generated weights and values lists

if __name__ == "__main__":
    size = 7  # Number of items
    capacity = 9  # Maximum Capacity of the knapsack
    weights, values = generate_knapsack_data(size,capacity)  # Generate Knapsack data
    values, weights = Sort(values, weights)  # Sort in descending order based on value-to-weight ratio

    start_time = time.time()  # Record start time of the algorithm
    greedy_cost, greedy_solution= knapsack_greedy(weights, values, capacity)  # Calculate optimal Knapsack cost using Greedy algorithm
    greedy_time = time.time() - start_time  # Calculate algorithm execution time

    print("Greedy Algorithm Cost:", greedy_cost)  # Print optimal Knapsack cost using Greedy algorithm
    print("Greedy Algorithm Solution:", greedy_solution)  # Print optimal Knapsack solution using Greedy algorithm
    print("Greedy Algorithm Time:", greedy_time)  # Print algorithm execution time