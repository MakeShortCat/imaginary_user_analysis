import numpy as np
from scipy.stats import beta
import random
from itertools import combinations
import pandas as pd

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

# Let's combine all Q-tables into a big numpy array
q_tables = np.array([agent.Q for agent in agents])

# Now let's save this big array into a .npy file
np.save('q_tables_+2_arr0.2_e0.1_least4000_epochange.npy', q_tables)

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
random_integer = random.randint(12, 10000)

df.to_csv(f'C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/nonupdate/user_simulation_beta+2_arr0.2_e0.1_winlog_noupdate{random_integer}.csv', index_label=False)

print('done')

