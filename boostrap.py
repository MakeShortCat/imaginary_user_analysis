import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

Fth_added = pd.read_csv('simple_test/user_simulation_4th_+2_arr0.2_least4000_rate0.01.csv')

original = pd.read_csv('simple_test/user_simulation_beta+2_arr0.2_e01_least4000_rate_0.01_epo100.csv')

Fth_added['action_history'] = Fth_added['action_history'].apply(eval)

original['action_history'] = original['action_history'].apply(eval)

def fix_stratagy(df):
    for i,j in enumerate(df['stratagy']):
        numbers_str = re.findall(r'array\(\[(.*?)\], dtype=int64\)', j)[1]  # 숫자를 찾습니다.
        df['stratagy'].iloc[i] = [int(num) for num in numbers_str.split(',')] 

    df['Win_rate'] = df['Wins']/(df['Losses']+df['Wins'])

    df['easy'] = df['stratagy'].apply(lambda x: x[0])

    df['normal'] = df['stratagy'].apply(lambda x: x[1])

    df['hard'] = df['stratagy'].apply(lambda x: x[2]) 

    df["total_match"]= df['Wins'] + df['Losses']

    df['easy_ratio'] = df['easy']/df["total_match"]

    df['normal_ratio'] = df['normal']/df["total_match"]

    df['hard_ratio'] = df['hard']/df["total_match"]



fix_stratagy(original)
fix_stratagy(Fth_added)

Fth_added['new'] = Fth_added['stratagy'].apply(lambda x: x[3])
Fth_added['new_ratio'] = Fth_added['new']/Fth_added['total_match']

import matplotlib.pyplot as plt
import numpy as np

division = ["easy", "normal", 'hard']

means = {
    'original': (original['easy_ratio'].mean(), original['normal_ratio'].mean(), original['hard_ratio'].mean()),
    'Fth_added': (Fth_added['easy_ratio'].mean(), Fth_added['normal_ratio'].mean(), Fth_added['hard_ratio'].mean()),

}

x = np.arange(len(division))  # the label locations
width = 0.4  # the width of the bars
multiplier = 0.5

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

aa = ax.bar(3.2, Fth_added['new_ratio'].mean(), width, label = 'Fth_added_new')
ax.bar_label(aa, padding=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('all')
ax.set_xticks([0.4, 1.4, 2.4, 3.2], ["easy", "normal", 'hard', 'new'])
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 0.5)

plt.show()










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
from sklearn.utils import resample

def calculate_entropy(lst):
    # 각 요소의 빈도를 저장하는 딕셔너리
    frequency_dict = {i: lst.count(i) for i in set(lst)}
    
    # 리스트의 길이
    lst_length = len(lst)
    
    # 엔트로피를 계산
    entropy = -sum((frequency / lst_length) * math.log2(frequency / lst_length) 
                   for frequency in frequency_dict.values())
    
    return entropy

def action_history(df):

    df['run_length'] = df['action_history'].apply(lambda x : calculate_run_lengths(x))

    df['entropy'] = df['action_history'].apply(lambda x : calculate_entropy(x))

    df['max_run'] = df['run_length'].apply(lambda x : max(x))

    df['mean_run'] = df['run_length'].apply(lambda x : sum(x)/len(x))

    df['run_over10'] = df['run_length'].apply(lambda x : sum(1 for num in x if num > 10))

action_history(original)
action_history(Fth_added)

Fth_added.drop('Unnamed: 0', axis=1, inplace=True)

original['devided_entropy'] = original['entropy']/3

Fth_added['devided_entropy'] = Fth_added['entropy']/4

concated_df = pd.concat([original, Fth_added])

# 두 샘플의 차이 계산
max_run_observed_difference = np.mean(original['max_run']) - np.mean(Fth_added['max_run'])

mean_run_observed_difference = np.mean(original['mean_run']) - np.mean(Fth_added['mean_run'])

run_over10_observed_difference = np.mean(original['run_over10']) - np.mean(Fth_added['run_over10'])

devided_entropy_observed_difference = np.mean(original['devided_entropy']) - np.mean(Fth_added['devided_entropy'])

def bootstrap_data(concated_data, data1, data2, observed_difference):
    # 부트스트랩 반복
    n_iterations = 10000
    bootstrap_diffs = []
    for _ in range(n_iterations):
        # 합쳐진 데이터에서 재표본추출하여 두 그룹 생성
        bootstrap_combined = resample(concated_data, replace=True)
        bootstrap_a = bootstrap_combined[:len(data1)]
        bootstrap_b = bootstrap_combined[len(data2):]
        bootstrap_diff = np.mean(bootstrap_a) - np.mean(bootstrap_b)
        bootstrap_diffs.append(bootstrap_diff)

    # 신뢰구간 계산 (95% 신뢰구간)
    confidence_interval = [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]

    # 관찰된 차이가 신뢰구간에 있는지 확인
    if confidence_interval[0] <= observed_difference<= confidence_interval[1]:
        print(0)
    else:
        print(1)

    return [bootstrap_diffs, observed_difference]

qwer = bootstrap_data(concated_df['max_run'], original['max_run'], Fth_added['max_run'], max_run_observed_difference)

bootstrap_data(concated_df['mean_run'], original['mean_run'], Fth_added['mean_run'], max_run_observed_difference)

bootstrap_data(concated_df['run_over10'], original['run_over10'], Fth_added['run_over10'], max_run_observed_difference)

bootstrap_data(concated_df['devided_entropy'], original['devided_entropy'], Fth_added['devided_entropy'], max_run_observed_difference)


# 전체 승률 그래프 그리기
plt.hist(qwer[0], bins=100, edgecolor='black')
plt.xlabel('X-value')
plt.ylabel('Frequency')
plt.axvline(qwer[1], color='r')
plt.title('Histogram')
plt.show()

# 부트스트랩 반복
n_iterations = 10000
bootstrap_diffs = []
for _ in range(n_iterations):
    bootstrap_a = resample(original['mean_run'], replace=True)
    bootstrap_b = resample(Fth_added['mean_run'], replace=True)
    bootstrap_diff = np.mean(bootstrap_a) - np.mean(bootstrap_b)
    bootstrap_diffs.append(bootstrap_diff)

# 신뢰구간 계산 (95% 신뢰구간)
confidence_interval = [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]

# 관찰된 차이가 신뢰구간에 있는지 확인
if confidence_interval[0] <= mean_run_observed_difference <= confidence_interval[1]:
    print("The difference between the two designs is not significant at the 95% confidence level.")
else:
    print("The difference between the two designs is significant at the 95% confidence level.")

# 전체 승률 그래프 그리기
plt.hist(bootstrap_diffs, bins=100, edgecolor='black')
plt.xlabel('X-value')
plt.ylabel('Frequency')
plt.axvline(mean_run_observed_difference, color='r')
plt.title('Histogram')
plt.show()


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

effect_size = cohen_d(original['devided_entropy'], Fth_added['devided_entropy'])

print("Effect size (Cohen's d): ", effect_size)


# original data
data = [0.1, 0.2, 0.3, 0.4, 0.5]

# resampled data
resampled_data = resample(data, replace=True)

resampled_data


import matplotlib.pyplot as plt

design1 = original['devided_entropy']
design2 = Fth_added['devided_entropy']

plt.boxplot([design1, design2], labels=['original', 'Fth_added'])
plt.title('Box plot of original devided_entropy and Fth_added devided_entropy')
plt.ylabel('Values')
plt.show()

import numpy as np
from scipy.stats import mannwhitneyu

# 예시 데이터
design1 = original['max_run']
design2 = Fth_added['max_run']

# 맨-휘트니 U 검정 수행
result = mannwhitneyu(design1, design2)

print(f'U-statistic: {result.statistic}')
print(f'p-value: {result.pvalue}')



