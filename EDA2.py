import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e0.1_least4000_epochange.csv')

for i,j in enumerate(df['stratagy']):
    numbers_str = re.findall(r'array\(\[(.*?)\], dtype=int64\)', j)[1]  # 숫자를 찾습니다.
    df['stratagy'].iloc[i] = [int(num) for num in numbers_str.split(',')] 

df['stratagy'] = df['stratagy'].apply(eval)

df['Win_rate'] = df['Wins']/(df['Losses']+df['Wins'])

df['easy'] = df['stratagy'].apply(lambda x: x[0])

df['normal'] = df['stratagy'].apply(lambda x: x[1])

df['hard'] = df['stratagy'].apply(lambda x: x[2]) 

df["total_match"]= df['Wins'] + df['Losses']

df['easy_ratio'] = df['easy']/df["total_match"]

df['normal_ratio'] = df['normal']/df["total_match"]

df['hard_ratio'] = df['hard']/df["total_match"]

df['action_history'] = df['action_history'].apply(eval)


def calculate_run_lengths(lst):
    run_lengths = []  # 각 런의 길이를 저장할 리스트
    current_run_length = 1  # 현재 런의 길이

    # 리스트의 모든 요소를 반복
    for i in range(1, len(lst)):
        # 만약 현재 요소가 이전 요소와 같다면
        if lst[i] == lst[i - 1]:
            # 현재 런의 길이를 증가
            current_run_length += 1
        else:
            # 아니라면, 현재 런을 종료하고 새로운 런을 시작
            run_lengths.append(current_run_length)
            current_run_length = 1

    # 마지막 런의 길이를 추가
    run_lengths.append(current_run_length)

    return run_lengths

import math

def calculate_entropy(lst):
    # 각 요소의 빈도를 저장하는 딕셔너리
    frequency_dict = {i: lst.count(i) for i in set(lst)}
    
    # 리스트의 길이
    lst_length = len(lst)
    
    # 엔트로피를 계산
    entropy = -sum((frequency / lst_length) * math.log2(frequency / lst_length) 
                   for frequency in frequency_dict.values())
    
    return entropy



df['run_length'] = df['action_history'].apply(lambda x : calculate_run_lengths(x))

df['entropy'] = df['action_history'].apply(lambda x : calculate_entropy(x))

df['max_run'] = df['run_length'].apply(lambda x : max(x))

df['mean_run'] = df['run_length'].apply(lambda x : sum(x)/len(x))

df['run_over10'] = df['run_length'].apply(lambda x : sum(1 for num in x if num > 10))

df['run_over10_div'] = df['run_over10']/df['total_match']

df.reset_index(inplace=True, drop=True)

df.describe()

df.columns


# 전체 승률 그래프 그리기
plt.hist(df['Win_rate'], bins=150, edgecolor='black')
plt.xlabel('X-value')
plt.xticks(np.arange(0.4,0.6, 0.01))
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

list_values = df['action_history']
# 결과를 저장할 빈 dictionary 생성
values_dict = {}

for row in range(len(list_values)):
    # 각 row에 대해 결과를 저장할 빈 리스트 생성
    values_0 = []
    values_1 = []
    values_2 = []

    count_0 = 0
    count_1 = 0
    count_2 = 0

    for i in range(len(list_values[row])):
        if list_values[row][i] == 0:
            count_0 += 1
        elif list_values[row][i] == 1:
            count_1 += 1
        elif list_values[row][i] == 2:
            count_2 += 1

        values_0.append(count_0)
        values_1.append(count_1)
        values_2.append(count_2)

    # 결과를 dictionary에 저장
    values_dict[row] = {'values_0': values_0, 'values_1': values_1, 'values_2': values_2}

for i in range(1024):
    values_dict[i]['values_0']
    values_dict[i]['values_1']
    values_dict[i]['values_2']

# 각 행에 대한 누적 값을 저장할 리스트 초기화
total_values_0 = [0] * 20001
total_values_1 = [0] * 20001
total_values_2 = [0] * 20001

for row in range(len(list_values)):
    count_0 = 0
    count_1 = 0
    count_2 = 0

    for i in range(min(len(list_values[row]), 20001)):
        if list_values[row][i] == 0:
            count_0 += 1
        elif list_values[row][i] == 1:
            count_1 += 1
        elif list_values[row][i] == 2:
            count_2 += 1

        total_values_0[i] += count_0
        total_values_1[i] += count_1
        total_values_2[i] += count_2


for k in range(600,610):
    # 결과를 저장할 빈 리스트 생성
    values_0 = []
    values_1 = []
    values_2 = []


    for i in range(len(list_values[k])):
        values_0.append(sum(1 for j in list_values[k][:i] if j == 0))
        values_1.append(sum(1 for j in list_values[k][:i] if j == 1))
        values_2.append(sum(1 for j in list_values[k][:i] if j == 2))

    # 결과를 그래프로 표현
    plt.figure(figsize=(12, 8))

    plt.ylabel('select_count')
    plt.xlabel('total_count')

    plt.subplot(1, 1, 1)
    plt.plot(values_0, label='easy')
    plt.title('easy')
    plt.grid(True)

    plt.subplot(1, 1, 1)
    plt.plot(values_1, label='normal', color='red')
    plt.title('normal')
    plt.grid(True)

    plt.subplot(1, 1, 1)
    plt.plot(values_2, label='hard', color='green')
    plt.title('hard')
    plt.grid(True)

    plt.legend(fontsize=18)

    plt.tight_layout()
    plt.show()

df.describe()

df.columns