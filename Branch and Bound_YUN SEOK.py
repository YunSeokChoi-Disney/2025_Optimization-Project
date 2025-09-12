from queue import PriorityQueue

class Item:
    def __init__(self, weight, value):
        self.weight = weight  # Initialize the weight of the item
        self.value = value  # Initialize the value of the item

    def __repr__(self):
        return f"Item(weight={self.weight}, value={self.value})"  # String representation for easy debugging

class Node:
    def __init__(self, level, profit, weight, contains):
        self.level = level  # Current level in decision tree (index in items list)
        self.profit = profit  # Cumulative profit up to this node
        self.weight = weight  # Cumulative weight up to this node
        self.contains = contains  # List of items included in the path to this node

    def __lt__(self, other):
        return other.profit - self.profit  # Define less than for priority queue based on profit

def custom_sort(arr):
    n = len(arr)  # Length of the array
    for i in range(n):
        max_index = i  # Assume the element at i is the maximum
        for j in range(i+1, n):  # Iterate over the array starting from i+1 to the end
            if arr[j].value / arr[j].weight > arr[max_index].value / arr[max_index].weight:
                max_index = j  # Update max_index if a larger ratio is found
        arr[i], arr[max_index] = arr[max_index], arr[i]  # Swap the found maximum with the current element

def bound(u, n, W, arr):
    if u.weight >= W:
        return 0  # Cannot add more items if weight is equal or over the limit

    profit_bound = u.profit  # Initialize bound with the current node's profit
    j = u.level + 1  # Start checking items after the current item
    total_weight = u.weight  # Start with current total weight

    # Add full items as long as weight limit is not exceeded
    while j < n and total_weight + arr[j].weight <= W:
        total_weight += arr[j].weight
        profit_bound += arr[j].value
        j += 1

    # Add fractional part of the next item if weight limit is not yet reached
    if j < n:
        profit_bound += int((W - total_weight) * arr[j].value / arr[j].weight)

    return profit_bound  # Return the estimated profit bound

def knapsack(W, arr, n):
    custom_sort(arr)  # Sort items by descending value/weight ratio using custom sort
    priority_queue = PriorityQueue()  # Create a priority queue
    u = Node(-1, 0, 0, [])  # Initialize with a dummy node at level -1
    priority_queue.put(u)  # Add dummy node to the queue

    max_profit = 0  # Initialize max profit as 0
    best_items = []  # Initialize best items list

    while not priority_queue.empty():
        u = priority_queue.get()  # Retrieve node with highest profit from the queue

        if u.level == n - 1:
            continue  # If we've considered all items, continue

        # Node for including the next item
        v = Node(u.level + 1, u.profit + arr[u.level + 1].value, u.weight + arr[u.level + 1].weight, u.contains + [arr[u.level + 1]])
        if v.weight <= W:
            if v.profit > max_profit:
                max_profit = v.profit  # Update max profit if current profit is higher
                best_items = v.contains  # Update list of best items
            v_bound = bound(v, n, W, arr)  # Calculate profit bound for v
            if v_bound > max_profit:
                priority_queue.put(v)  # Add node to the queue if bound is higher than max profit

        # Node for not including the next item
        v = Node(u.level + 1, u.profit, u.weight, u.contains)
        v_bound = bound(v, n, W, arr)  # Calculate profit bound for v
        if v_bound > max_profit:
            priority_queue.put(v)  # Add node to the queue if bound is higher than max profit

    return max_profit, best_items  # Return the maximum profit and the best items found

# Driver program to test the above function
W = 15  # Maximum allowable weight for the knapsack
arr = [
    Item(12, 4),
    Item(1, 2),
    Item(4, 10),
    Item(1, 1),
    Item(2, 2)
]  # List of items to consider
n = len(arr)  # Number of items

max_profit, best_items = knapsack(W, arr, n)  # Execute knapsack function
print("Maximum possible profit =", max_profit)  # Print maximum profit achieved
print("Items in the optimal knapsack:", best_items)  # Print items that are included in the optimal solution