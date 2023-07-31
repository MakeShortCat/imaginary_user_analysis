import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e01_least4000_rate_0.01_epo100.csv')


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

# df.to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+1.5_arr0.2_e0.1_least4000_rate0.015_EDA.csv')

# df = pd.read_csv('simple_test/user_simulation_beta+1.5_arr0.2_e0.1_least4000_rate0.015_EDA.csv')


# 전체 승률 그래프 그리기
plt.hist(df['Win_rate'], bins=150, edgecolor='black')
plt.xlabel('X-value')
plt.xticks(np.arange(0.4,0.6, 0.01))
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

df_sorted = df.sort_values('Win_rate').reset_index()

low_winrate = df_sorted[df_sorted['Win_rate'] <= 0.48]

midlow_winrate = df_sorted[(df_sorted['Win_rate'] >= 0.48) & (df_sorted['Win_rate']<=0.5)]

midhigh_winrate = df_sorted[(df_sorted['Win_rate'] >= 0.5) & (df_sorted['Win_rate']<=0.52)]

high_winrate = df_sorted[df_sorted['Win_rate'] >= 0.52]

def winrate_plot(df, name):

    total_match = df['Wins'] + df['Losses']

    first_elements = df['easy']/total_match

    second_elements = df['normal']/total_match

    third_elements = df['hard']/total_match

    new_elements = df['new']/total_match

    plt.bar(x =['easy', 'normal', 'hard', 'new'], height = [first_elements.sum()/len(df), second_elements.sum()/len(df), third_elements.sum()/len(df), new_elements.sum()/len(df)], edgecolor='black')
    # plt.bar(x =['easy', 'normal', 'hard'], height = [first_elements.sum()/len(df), second_elements.sum()/len(df), third_elements.sum()/len(df)], edgecolor='black')

    plt.xlabel('stratagy')
    plt.ylabel('Frequency')
    plt.title(f'{name}')
    plt.show()

# 파이플롯으로 수정할 것
winrate_plot(low_winrate, 'low_winrate')

winrate_plot(midlow_winrate,'midlow_winrate')

winrate_plot(midhigh_winrate,'midhigh_winrate')

winrate_plot(high_winrate,'high_winrate')

winrate_plot(df, 'all')

from scipy.stats import beta
def plot_beta_distribution_3(a, b):
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, a, b) + 2

    log_list = []

    for i in range(1000):
        if i == 500:
            pass
        else:
            log_list.append(y[i]*(i-500))
    x = np.linspace(0, 1, 999)

    plt.plot(x, log_list, lw=2)
    plt.title(f'{round(a, 1), round(b, 1)}')
    plt.xlabel('x')
    plt.axvline(x=0.5, color='r', linestyle='--')
    plt.ylabel('battle_power')
    plt.grid(True)
    plt.show()

low_easy = low_winrate.loc[low_winrate['easy_ratio'] == low_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

low_hard = low_winrate.loc[low_winrate['hard_ratio'] == low_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midlow_easy = midlow_winrate.loc[midlow_winrate['easy_ratio'] == midlow_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midlow_normal = midlow_winrate.loc[midlow_winrate['normal_ratio'] == midlow_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midlow_hard = midlow_winrate.loc[midlow_winrate['hard_ratio'] == midlow_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midhigh_easy = midhigh_winrate.loc[midhigh_winrate['easy_ratio'] == midhigh_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midhigh_normal = midhigh_winrate.loc[midhigh_winrate['normal_ratio'] == midhigh_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

midhigh_hard = midhigh_winrate.loc[midhigh_winrate['hard_ratio'] == midhigh_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

high_easy = high_winrate.loc[high_winrate['easy_ratio'] == high_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

high_normal = high_winrate.loc[high_winrate['normal_ratio'] == high_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

high_hard = high_winrate.loc[high_winrate['hard_ratio'] == high_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

low_winrate[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)

new_st = df.loc[df['new_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio', 'new_ratio']].max(axis=1)]




# 특정 구간 그래프 모양 살피기
for a,b in zip(low_easy['alpha'][35:45], low_easy['beta'][35:45]):
        plot_beta_distribution_3(a, b)

# 특정 구간 그래프 모양 살피기
for a,b in zip(low_hard['alpha'], low_hard['beta']):
        plot_beta_distribution_3(a, b)


###########################
for a,b in zip(midlow_easy['alpha'][50:70], midlow_easy['beta'][50:70]):
        plot_beta_distribution_3(a, b)

for a,b in zip(midlow_normal['alpha'], midlow_normal['beta']):
        plot_beta_distribution_3(a, b)

for a,b in zip(midlow_hard['alpha'][100:120], midlow_hard['beta'][100:120]):
        plot_beta_distribution_3(a, b)


################################
for a,b in zip(midhigh_easy['alpha'], midhigh_easy['beta']):
        plot_beta_distribution_3(a, b)

for a,b in zip(midhigh_normal['alpha'][110:120], midhigh_normal['beta'][110:120]):
        plot_beta_distribution_3(a, b)

for a,b in zip(midhigh_hard['alpha'][30:40], midhigh_hard['beta'][30:40]):
        plot_beta_distribution_3(a, b)


############################
for a,b in zip(high_normal['alpha'][100:110], high_normal['beta'][100:110]):
        plot_beta_distribution_3(a, b)

for a,b in zip(high_hard['alpha'], high_hard['beta']):
        plot_beta_distribution_3(a, b)


df['action_history'] = df['action_history'].apply(eval)

df['action_history']

df.loc[49]


list_values = df['action_history']

df.describe()

# 결과를 저장할 빈 리스트 생성
values_0 = []
values_1 = []
values_2 = []
values_3 = []

# 리스트의 각 값에 대해 최대값, 평균값, 10보다 큰 값의 개수를 계산
for i in range(1, 4000):
    values_0.append(sum(1 for j in list_values[49][:i] if j == 0))
    values_1.append(sum(1 for j in list_values[49][:i] if j == 1))
    values_2.append(sum(1 for j in list_values[49][:i] if j == 2))
    values_3.append(sum(1 for j in list_values[49][:i] if j == 3))

# 결과를 그래프로 표현
plt.figure(figsize=(12, 8))

plt.ylabel('select_count')
plt.xlabel('total_count')

plt.subplot(1, 1, 3)
plt.plot(values_0, label='easy')
plt.title('easy')
plt.grid(True)

plt.subplot(1, 1, 3)
plt.plot(values_1, label='normal', color='red')
plt.title('normal')
plt.grid(True)

plt.subplot(1, 1, 3)
plt.plot(values_2, label='hard', color='green')
plt.title('hard')
plt.grid(True)

plt.subplot(1, 1, 3)
plt.plot(values_3, label='new', color='y')
plt.title('new')
plt.grid(True)

plt.legend(fontsize=18)

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import numpy as np

# 예를 들어, 0에서 50까지 무작위 정수 300개로 구성된 리스트를 생성
list_values = df['run_length'].iloc[964]

# 결과를 저장할 빈 리스트 생성
max_values = []
mean_values = []
count_values = []
entropy_val = []

# 리스트의 각 값에 대해 최대값, 평균값, 10보다 큰 값의 개수를 계산
for i in range(1, len(list_values) + 1):
    max_values.append(max(list_values[:i]))
    mean_values.append(np.mean(list_values[:i]))
    count_values.append(sum(j > 10 for j in list_values[:i]))

# 결과를 그래프로 표현
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(max_values, label='Max value')
plt.title('Max Value')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(mean_values, label='Average value', color='red')
plt.title('Average Value')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(count_values, label='Count of values > 10', color='green')
plt.title('Count of Values > 10')
plt.grid(True)

plt.tight_layout()
plt.show()

df['total_match'].mean()

df['max_run'].describe()

df['mean_run'].describe()

df['run_over10'].describe()

df.describe()

easy_st = df.loc[df['easy_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

normal_st = df.loc[df['normal_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

hard_st = df.loc[df['hard_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio']].max(axis=1)]

df.loc[df['hard_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio', 'new_ratio']].max(axis=1)]


np.array([easy_st['max_run'].mean(), normal_st['max_run'].mean(), hard_st['max_run'].mean()]).var()

easy_st_samples = np.random.choice(df['max_run'], size=(10000, len(easy_st['max_run'])), replace=True)

normal_st_samples = np.random.choice(df['max_run'], size=(10000, len(normal_st['max_run'])), replace=True)

hard_st_samples = np.random.choice(df['max_run'], size=(10000, len(hard_st['max_run'])), replace=True)

np.array([easy_st['max_run'].mean(), normal_st['max_run'].mean(), hard_st['max_run'].mean()]).var()

max_run_var_list = []
for i in range(10000):
     max_run_var_list.append(np.array([easy_st_samples[i].mean(), normal_st_samples[i].mean(), hard_st_samples[i].mean()]).var())

above = np.sum(np.array(max_run_var_list) >= 3.3165) / len(max_run_var_list)

above

easy_st_samples = np.random.choice(df['mean_run'], size=(1000000, len(easy_st['mean_run'])), replace=True)

normal_st_samples = np.random.choice(df['mean_run'], size=(1000000, len(normal_st['mean_run'])), replace=True)

hard_st_samples = np.random.choice(df['mean_run'], size=(1000000, len(hard_st['mean_run'])), replace=True)


max_run_avg_list = []
for i in range(1000000):
     max_run_avg_list.append(np.array([easy_st_samples[i].mean(), normal_st_samples[i].mean(), hard_st_samples[i].mean()]).var())

np.array([easy_st['mean_run'].mean(), normal_st['mean_run'].mean(), hard_st['mean_run'].mean()]).var()

above = np.sum(np.array(max_run_avg_list) >= 0.0017221375509630132) / len(max_run_avg_list)

easy_st_samples = np.random.choice(df['run_over10'], size=(10000, len(easy_st['run_over10'])), replace=True)

normal_st_samples = np.random.choice(df['run_over10'], size=(10000, len(normal_st['run_over10'])), replace=True)

hard_st_samples = np.random.choice(df['run_over10'], size=(10000, len(hard_st['run_over10'])), replace=True)


run_over10_list = []
for i in range(10000):
     run_over10_list.append(np.array([easy_st_samples[i].mean(), normal_st_samples[i].mean(), hard_st_samples[i].mean()]).var())

np.array([easy_st['run_over10'].mean(), normal_st['run_over10'].mean(), hard_st['run_over10'].mean()]).var()

above = np.sum(np.array(run_over10_list) >= 4235.696329584653) / len(run_over10_list)

above

# 히스토그램 그리기
plt.hist(run_over10_list, bins=30, edgecolor='black')
plt.xlabel('var_sample')
plt.ylabel('Frequency')
plt.axvline(4235, color = 'r')
plt.title('Histogram of Bootstrap Sample')
plt.show()

df.describe()

easy_st.describe()

normal_st.describe()

hard_st.describe()


# 부트스트랩 샘플링으로 표본 평균 계산
bootstrap_samples = np.random.choice(df['max_run'], size=(10000, len(easy_st)), replace=True)
bootstrap_sample_means = np.mean(bootstrap_samples, axis=1)


# 히스토그램 그리기
plt.hist(bootstrap_sample_means, bins=30, edgecolor='black')
plt.xlabel('Average Max Streak')
plt.ylabel('Frequency')
plt.title('Histogram of Bootstrap Sample Means')
plt.show()



import numpy as np

# 부트스트랩 샘플을 추출하는 함수
def bootstrap_sample(data, n_bootstrap):
    return np.random.choice(data, size=len(data)*n_bootstrap, replace=True)

# 부트스트랩 샘플의 평균을 계산하는 함수
def bootstrap_mean(data, n_bootstrap):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample_data = bootstrap_sample(data, 1)
        bootstrap_means.append(np.mean(bootstrap_sample_data))
    return bootstrap_means

n_bootstrap = 10000  # 부트스트랩 샘플 수

# 각 표본에 대해 부트스트랩 평균을 계산
bootstrap_means1 = bootstrap_mean(easy_st['run_over10'], n_bootstrap)
bootstrap_means2 = bootstrap_mean(normal_st['run_over10'], n_bootstrap)
bootstrap_means3 = bootstrap_mean(hard_st['run_over10'], n_bootstrap)

# 부트스트랩 평균의 차이 계산
diff_means12 = np.subtract(bootstrap_means1, bootstrap_means2)
diff_means13 = np.subtract(bootstrap_means1, bootstrap_means3)
diff_means23 = np.subtract(bootstrap_means2, bootstrap_means3)

# 차이가 통계적으로 유의미한지 확인 (0이 신뢰 구간에 포함되는지 확인)
ci = [2.5, 97.5]  # 95% 신뢰 구간

ci_diff_means12 = np.percentile(diff_means12, ci)
ci_diff_means13 = np.percentile(diff_means13, ci)
ci_diff_means23 = np.percentile(diff_means23, ci)

print(f"Sample1과 Sample2의 평균 차이의 95% 신뢰 구간: {ci_diff_means12}")
print(f"Sample1과 Sample3의 평균 차이의 95% 신뢰 구간: {ci_diff_means13}")
print(f"Sample2과 Sample3의 평균 차이의 95% 신뢰 구간: {ci_diff_means23}")


df['Win_rate'].var()

