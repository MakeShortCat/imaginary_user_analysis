import numpy as np
from scipy.stats import beta
import random
from itertools import combinations
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score

hyperparameters = {
    'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}]
}

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_beta+2_arr0.2_e0.1_winlog_final.csv')

df.drop(['cluster'], axis=1, inplace=True)

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
    train['ccc'] = temp_df['ccc']

    predictor = TabularPredictor(label='ccc',problem_type= 'multiclass',
                                  eval_metric = 'accuracy').fit(train, presets='best_quality',
                                save_space=True, hyperparameters=hyperparameters, num_bag_sets=4, time_limit=70,num_cpus=8 ,fit_weighted_ensemble=False)
    
    return predictor


predictor_200 = make_predictor(200)

predictor = predictor_200

class Agent:
    def __init__(self, a, b, epsilon=0.1, learning_rate=0.4, discount_factor=0.70, agent_id=None, special_agent=False):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.zeros(3)
        self.action_history = []
        self.winlog = []
        self.wins = 0
        self.losses = 0
        self.agent_id = agent_id
        self.special_agent = special_agent
        self.consecutive_losses = 0


    def choose_action(self):
        if self.special_agent and self.consecutive_losses >= 3 and self.wins + self.losses > 200:
            list_temp = self.special_strategy()
            self.consecutive_losses = 0
            print(list_temp)
            return list_temp

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2])  # Explore: select a random action
        else:
            return np.argmax(self.Q)  # Exploit: select the action with max value (greedy)

    def update(self, action, reward, next_action):
        self.Q[action] += self.learning_rate * (reward + self.discount_factor * self.Q[next_action] - self.Q[action])
        if reward < 0: 
            self.consecutive_losses += 1
            if self.special_agent:
                self.b +=1
        else:
            self.consecutive_losses = 0
            if self.special_agent:
                self.a +=1


    def special_strategy(self):
        global predictor
        temp_list = np.array(self.action_history)
        temp_win = np.array(self.winlog)

        # 배열에서 값이 n인 요소들의 인덱스를 가져옴
        st0_loc = np.where(temp_list == 0)[0]
        st1_loc = np.where(temp_list == 1)[0]
        st2_loc = np.where(temp_list == 2)[0]

        # 전략별 승률 계산
        WL0 = temp_win[st0_loc[0]].mean()
        WL1 = temp_win[st1_loc[0]].mean()
        WL2 = temp_win[st2_loc[0]].mean()

        easy_ratio = np.where(temp_list == 0)[0].shape[0] / len(temp_list)
        normal_ratio = np.where(temp_list == 1)[0].shape[0] / len(temp_list)
        hard_ratio = np.where(temp_list == 2)[0].shape[0] / len(temp_list)
        ab_ratio = self.wins / self.losses

        pridict_df = pd.DataFrame([[WL0,	WL1,	WL2,	easy_ratio,	normal_ratio,	hard_ratio, ab_ratio]],
                                   columns=['easy_winrate', 'normal_winrate', 'hard_winrate', 'easy_ratio_200', 'normal_ratio_200', 'hard_ratio_200', 'ab_ratio'])
        
        if self.wins + self.losses < 200:
            prediction = predictor_200.predict(pridict_df)
            predictor = predictor_200
            return prediction[0]

        elif accuracy_score(df['ccc'], predictor.predict(cal_hist(self.wins + self.losses).iloc[:, 16:23])) < 0.8:
            print(accuracy_score(df['ccc'], predictor.predict(cal_hist(self.wins + self.losses).iloc[:, 16:23])), self.wins + self.losses)
            predictor = make_predictor(self.wins + self.losses)
             
            
        prediction = predictor.predict(pridict_df)

        return prediction[0]

def simulate(agents, epochs=70):
    for _ in range(epochs):
        comdd = list(combinations(agents, 2))
        random.shuffle(comdd)
        for agent1, agent2 in comdd:  # Simulate a match between every pair of agent
            if agent1.wins > 1000 and agent2.wins > 1000 and abs(agent1.wins/(agent1.wins + agent1.losses + 1) - agent2.wins/(agent2.wins + agent2.losses + 1)) > 0.01:
                continue

            action1 = agent1.choose_action()
            action2 = agent2.choose_action()

            # Record the chosen actions
            agent1.action_history.append(action1)
            agent2.action_history.append(action2)

            random_value_hard = np.random.uniform(0, 0.1) if np.random.rand() < 0.5 else np.random.uniform(0.9, 1)
            random_value_normal = np.random.uniform(0.1, 0.3) if np.random.rand() < 0.5 else np.random.uniform(0.7, 0.9)
            random_value_easy = np.random.uniform(0.3, 0.5) if np.random.rand() < 0.5 else np.random.uniform(0.5, 0.7)

            x = [random_value_easy, random_value_normal, random_value_hard][action1]

            random_value_hard = np.random.uniform(0, 0.1) if np.random.rand() < 0.5 else np.random.uniform(0.9, 1)
            random_value_normal = np.random.uniform(0.1, 0.3) if np.random.rand() < 0.5 else np.random.uniform(0.7, 0.9)
            random_value_easy = np.random.uniform(0.3, 0.5) if np.random.rand() < 0.5 else np.random.uniform(0.5, 0.7)

            y = [random_value_easy, random_value_normal, random_value_hard][action2]
            
            rewards = beta.pdf([x, y], [agent1.a, agent2.a], [agent1.b, agent2.b]) + 2

            rewards = rewards * [(x - 0.5)*1000, (y - 0.5) * 1000]
            
            if rewards[0] > rewards[1]:
                reward1, reward2 = 1, -1
                agent1.wins += 1
                agent2.losses += 1
                agent1.winlog.append(1)
                agent2.winlog.append(0)

            elif rewards[0] < rewards[1]:
                reward1, reward2 = -1, 1
                agent1.losses += 1
                agent2.wins += 1
                agent1.winlog.append(0)
                agent2.winlog.append(1)

            else:
                reward1, reward2 = 0, 0

            next_action1 = agent1.choose_action()
            next_action2 = agent2.choose_action()

            agent1.update(action1, reward1, next_action1)
            agent2.update(action2, reward2, next_action2)
            

from itertools import product
array = np.arange(0.2, 6.6, 0.2)

# Initialize ten agents with different beta distributions
agents = [Agent(a=a, b=b) for a, b in list(product(array, repeat=2))]

agents.append(Agent(a=3, b=3, agent_id=0, special_agent=True))

# Simulate the game
simulate(agents)


# 기존의 데이터프레임
df = pd.DataFrame({'alpha':[1,2,3],
                   'beta' :[1,2,3],
                'Wins': [1,2,3],
                'Losses': [1,2,3],
                'stratagy' : [[1,2],[1,2],[1,2]],
                'action_history' : [1,2,3],
                'win_history' : [1,2,3]})

# Print out the action history and win/loss counts
for i, agent in enumerate(agents):
    # 추가할 행 데이터
    new_row = {'alpha': round(agent.a, 1) , 'beta': round(agent.b, 1),
            'Wins': agent.wins, 'Losses': agent.losses,
                'stratagy' : np.unique(agent.action_history, return_counts=True),
                'action_history' : agent.action_history, 'win_history' : agent.winlog}
    # 행 추가
    df.loc[len(df)] = new_row

df.drop([0,1,2], axis=0,inplace=True)

# 1부터 10000까지의 랜덤한 정수를 생성
random_integer = random.randint(0, 10000)

df.to_csv(f'C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/update/user_simulation_beta+2_arr0.2_e0.1_winlog_update{random_integer}.csv', index_label=False)

print('done')
