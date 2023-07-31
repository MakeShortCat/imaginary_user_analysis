import numpy as np
import matplotlib.pyplot as plt
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

    plt.plot(x, log_list, '-')
    plt.title(f'{round(a, 1), round(b, 1)}')
    plt.xlabel('x')
    # plt.axvline(x=0.3, color='r', linestyle='-')
    plt.axvline(x=0.5, color='r', linestyle='--')
    # plt.axvline(x=0.6, color='y', linestyle='-')
    # plt.axvline(x=0.8, color='y', linestyle='-')
    plt.ylabel('battle_power')
    plt.grid(True)
    plt.show()

a_values = np.arange(0.1, 3.2, 1)

plot_beta_distribution_3(2.1, 3.1)

for a in a_values:
    for b in a_values:
        plot_beta_distribution_3(a, b)


def plot_beta_distribution(a, b):
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, a, b)

    plt.plot(x, y, lw=2)
    plt.title(f'{round(a, 1), round(b, 1)}')
    plt.xlabel('x')
    plt.axvline(x=0.5, color='r', linestyle='--')
    # plt.axvline(x=0.16, color='g', linestyle='-')
    # plt.axvline(x=0.84, color='g', linestyle='-')
    # plt.axvline(x=0.33, color='y', linestyle='-')
    # plt.axvline(x=0.67, color='y', linestyle='-')

    plt.ylabel('Probability density')
    plt.grid(True)
    plt.show()

plot_beta_distribution(2.1, 3.1)

plot_beta_distribution(0.1,0.1)
plot_beta_distribution_3(0.1,0.1)

a_values = np.arange(0.1, 3.2, 1)

for a in a_values:
    for b in a_values:
        plot_beta_distribution(a, b)

import numpy as np
from scipy.stats import beta
from scipy.integrate import quad

alpha = 2.1
beta_ = 3.1
x1 = 0.16  # 구간의 시작
x2 = 0.33  # 구간의 끝

x_list = [0, 0.16, 0.33, 0.5, 0.67, 0.84, 1]

# 해당 구간에서의 기댓값을 계산하기 위한 함수 정의
def integrand(x):
    return x * beta.pdf(x, alpha, beta_)

for i in range(len(x_list)-1):
    # 적분하여 기댓값 계산
    expected_value, error = quad(integrand, x_list[i], x_list[i+1])
    print(expected_value)




expected_value, error = quad(integrand, x1, x2)
print(expected_value)


0.011902905838921441
0.0691920318507962
0.12082813923312853
0.12082813923312859
0.06919203185079616
0.01190290583892146
