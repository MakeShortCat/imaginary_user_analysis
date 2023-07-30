import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e0.1_winlog.csv')

for i,j in enumerate(df['stratagy']):
    numbers_str = re.findall(r'array\(\[(.*?)\], dtype=int64\)', j)[1]  # 숫자를 찾습니다.
    df['stratagy'].iloc[i] = [int(num) for num in numbers_str.split(',')] 

df['stratagy'] = df['stratagy'].apply(eval)

df['action_history'] = df['action_history'].apply(eval)

df['win_history'] = df['win_history'].apply(eval)

df['Win_rate'] = df['Wins']/(df['Losses']+df['Wins'])

df['easy'] = df['stratagy'].apply(lambda x: x[0])

df['normal'] = df['stratagy'].apply(lambda x: x[1])

df['hard'] = df['stratagy'].apply(lambda x: x[2]) 

df["total_match"]= df['Wins'] + df['Losses']

df['easy_ratio'] = df['easy']/df["total_match"]

df['normal_ratio'] = df['normal']/df["total_match"]

df['hard_ratio'] = df['hard']/df["total_match"]

clust_df = df.drop(['win_history','total_match','easy', 'normal', 'hard', 'action_history', 'stratagy', 'Wins', 'Losses', 'alpha','beta'], axis = 1)

# # 전체 승률 그래프 그리기
# plt.hist(df['Win_rate'], bins=150, edgecolor='black')
# plt.xlabel('X-value')
# plt.xticks(np.arange(0.4,0.6, 0.01))
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.show()

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

X = clust_df.iloc[:,1:]

# KMeans 클러스터링
k = 32  # 클러스터의 수
kmeans = KMeans(n_clusters=k, n_init = 240, random_state=21)
kmeans.fit(X)

# 클러스터 레이블 얻기
labels = kmeans.labels_

# 클러스터 중심점 얻기
centroids = kmeans.cluster_centers_

df['cluster'] = labels
df['ccc']=77
special_strategy_dict = {'easy_ratio': 0,'normal_ratio':1, 'hard_ratio': 2}

df.loc[df['cluster'] == 1].index

for i in range(32):
    selects = df.loc[df['cluster'] == i][['easy_ratio', 'normal_ratio', 'hard_ratio']].describe().iloc[1].idxmax()
    for j in df.loc[df['cluster'] == i].index:
        df['ccc'].loc[j] = special_strategy_dict[selects]



#cluster 0 : 승률이 낮고 hard 전략 사용
#cluster 1 : 승률이 높고 normal 전략 사용
#cluster 2 : 승률이 낮고 easy
#cluster 3 : 승률이 높고 normal과 전략 사용
#cluster 4 : 승률이 보통이고 골고루 전략 사용
#cluster 5 : 승률이 낮고 easy normal 전략 사용

# 결과 시각화
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
# plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='black', label='Centroids')
# plt.title("K-Means Clustering")
# plt.legend()
# plt.show()

# # 실루엣 점수 계산
# s_score = silhouette_score(X, labels)

# print(f"Silhouette Score: {s_score:.2f}")

# # 결과 시각화
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
# plt.title("K-Means Clustering")
# plt.show()

# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Elbow Method를 사용하여 최적의 클러스터 수 찾기
# wcss = []  # Within-Cluster-Sum-of-Squares
# K_range = range(1, 11)  # 1부터 10까지 클러스터 수를 테스트

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X)
#     # inertia_ 속성은 각 데이터 포인트에서 해당 클러스터 중심까지의 거리 제곱의 합을 나타냄
#     wcss.append(kmeans.inertia_)

# # 결과 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(K_range, wcss, marker='o', linestyle='--')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.xticks(K_range)
# plt.grid(True)
# plt.show()

df.to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e0.1_winlog_final.csv', index_label=False)


win_history = df['win_history']

EW1000 = []
NW1000 = []
HW1000 = []

easy_ratio_list = []
normal_ratio_list  = []
hard_ratio_list  = []


for i in range(1024):

    temp_list = np.array(df['action_history'].iloc[i][:400])

    # 배열에서 값이 n인 요소들의 인덱스를 가져옴
    st0_loc = np.where(temp_list == 0)[0]
    st1_loc = np.where(temp_list == 1)[0]
    st2_loc = np.where(temp_list == 2)[0]

    # 전략별 승률 계산
    WL0 = np.array(win_history.iloc[i])[[st0_loc]].mean()
    WL1 = np.array(win_history.iloc[i])[[st1_loc]].mean()
    WL2 = np.array(win_history.iloc[i])[[st2_loc]].mean()

    easy_ratio = np.where(temp_list == 0)[0].shape[0] / 100
    normal_ratio = np.where(temp_list == 1)[0].shape[0] / 100
    hard_ratio =np.where(temp_list == 2)[0].shape[0] / 100

    EW1000.append(WL0)
    NW1000.append(WL1)
    HW1000.append(WL2)

    easy_ratio_list.append(easy_ratio)
    normal_ratio_list.append(normal_ratio)
    hard_ratio_list.append(hard_ratio)

df['easy_winrate'] = EW1000
df['normal_winrate'] = NW1000
df['hard_winrate'] = HW1000

df['easy_ratio_200'] = easy_ratio_list
df['normal_ratio_200'] = normal_ratio_list
df['hard_ratio_200'] = hard_ratio_list

df['ab_ratio'] = df['alpha']/ df['beta']

import numpy as np

train = df.iloc[:, 16:23]
train['cluster'] = df['cluster']
hyperparameters = {
    'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}]
}
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='cluster',problem_type= 'multiclass', eval_metric = 'accuracy').fit(train, num_bag_sets=3, time_limit=30 ,presets='best_quality', save_space=True, num_gpus=1 , fit_weighted_ensemble=False)

predictor.leaderboard(silent=True) # best quality
