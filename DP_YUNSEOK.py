###Project2 DP for Knapsack Problem(2019048440 CHOI YUN SEOK)###
import numpy as np  # Import NumPy for array operations
import time  # Import time module for measuring execution time

#random seed
np.random.seed(42)      #for generating random, but fixed values for cost, weight


def Dynamic():
    start_time=time.time()
    global items
    dp = [[0] * (W + 1) for _ in range(n + 1)]  # dp[i][w] = i번째까지 고려, 무게 w일 때 최대 가치

    for i in range(1, n + 1):
        weight = items[i - 1][0]
        value = items[i - 1][1]
        for w in range(W + 1):
            if weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
            else:
                dp[i][w] = dp[i - 1][w]


    # 5. 선택된 아이템 역추적 - optimal solution에서 어떤 요소가 선택되었는지 파악 가능
    selected_items = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(items[i - 1])
            w -= items[i - 1][0]  # 무게만큼 줄이기

    # 6. 선택된 아이템 출력
    print('\n', '='*50)
    print('\n[Dynamic Programming 결과]')
    print("\n선택된 아이템 (정렬 후 번호. 무게, 가치):")
    C=0
    for item in reversed(selected_items):  # 선택 순서를 앞에서부터 보기 위해 뒤집음
        index_in_sorted = items.index(item) + 1
        A={item[0]}
        D=list(A)[0]
        C+=D
        print(f"{index_in_sorted}. ({item[0]}, {item[1]})")
        
    end_time= time.time() - start_time
    print('\n걸린 시간', end_time)
    print("\n최대 무게:", C)
    print("최대 가치:", dp[n][W])

