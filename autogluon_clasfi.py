import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e0.1_winlog_final.csv')

df['stratagy'] = df['stratagy'].apply(eval)

df['action_history'] = df['action_history'].apply(eval)

df['win_history'] = df['win_history'].apply(eval)

win_history = df['win_history']

def cal_hist(num):
    EW1000 = []
    NW1000 = []
    HW1000 = []

    easy_ratio_list = []
    normal_ratio_list  = []
    hard_ratio_list  = []

    for i in range(1024):

        temp_list = np.array(df['action_history'].iloc[i][:num])
        # 배열에서 값이 n인 요소들의 인덱스를 가져옴
        st0_loc = np.where(temp_list == 0)[0]
        st1_loc = np.where(temp_list == 1)[0]
        st2_loc = np.where(temp_list == 2)[0]

        # 전략별 승률 계산
        WL0 = np.array(win_history.iloc[i])[[st0_loc]].mean()
        WL1 = np.array(win_history.iloc[i])[[st1_loc]].mean()
        WL2 = np.array(win_history.iloc[i])[[st2_loc]].mean()

        easy_ratio = np.where(temp_list == 0)[0].shape[0] / num
        normal_ratio = np.where(temp_list == 1)[0].shape[0] / num
        hard_ratio =np.where(temp_list == 2)[0].shape[0] / num

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

    df['ab_ratio'] = df['alpha'] / df['beta']

    return df

def make_predictor(nums):
    temp_df = cal_hist(nums)
    train = temp_df.iloc[:, 16:23]
    train['cluster'] = temp_df['cluster']

    predictor = TabularPredictor(label='cluster',problem_type= 'multiclass', eval_metric = 'accuracy').fit(train, presets='best_quality', save_space=True, hyperparameters={'GBM':{}},num_bag_sets=1, time_limit=30, fit_weighted_ensemble=False)
    
    return predictor

pridict_df = pd.DataFrame([[1,1,1,1,1,1,1]],
                                   columns=['easy_winrate', 'normal_winrate', 'hard_winrate', 'easy_ratio_200', 'normal_ratio_200', 'hard_ratio_200', 'ab_ratio'])

aaa = make_predictor(200)
aaaa = aaa.predict(pridict_df)
aaaa[0]