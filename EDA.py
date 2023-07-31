import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/simple_test/user_simulation_4th_+2_arr0.2_least4000_rate0.01.csv')


for i,j in enumerate(df['stratagy']):
    numbers_str = re.findall(r'array\(\[(.*?)\], dtype=int64\)', j)[1]  # 숫자를 찾습니다.
    df['stratagy'].iloc[i] = [int(num) for num in numbers_str.split(',')] 

df['stratagy'] = df['stratagy'].apply(eval)

df['Win_rate'] = df['Wins']/(df['Losses']+df['Wins'])

df['easy'] = df['stratagy'].apply(lambda x: x[0])

df['normal'] = df['stratagy'].apply(lambda x: x[1])

df['hard'] = df['stratagy'].apply(lambda x: x[2]) 

df['new'] = df['stratagy'].apply(lambda x: x[3])

df["total_match"]= df['Wins'] + df['Losses']

df['easy_ratio'] = df['easy']/df["total_match"]

df['normal_ratio'] = df['normal']/df["total_match"]

df['hard_ratio'] = df['hard']/df["total_match"]

df['new_ratio'] = df['new']/df['total_match']


df['action_history'] = df['action_history'].apply(eval)

df

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


old_hard = np.array([ 34,  36,  64,  66,  99, 132, 160, 192, 200, 224, 225, 226, 232,
       233, 240, 256, 257, 258, 263, 265, 272, 273, 275, 288, 289, 290,
       303, 306, 307, 308, 309, 311, 312, 313, 320, 321, 322, 330, 331,
       335, 336, 337, 338, 339, 340, 341, 342, 344, 345, 346, 347, 348,
       350, 351, 352, 353, 366, 368, 369, 370, 371, 372, 373, 374, 375,
       376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 395, 399,
       401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
       414, 415, 416, 417, 432, 433, 434, 435, 436, 437, 438, 439, 440,
       441, 442, 443, 444, 445, 447, 448, 450, 451, 461, 462, 464, 465,
       467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
       480, 481, 482, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
       506, 507, 508, 509, 510, 511, 512, 513, 514, 527, 530, 531, 532,
       533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 546,
       561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573,
       574, 575, 576, 577, 578, 579, 595, 596, 597, 598, 599, 600, 601,
       602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 625, 627, 628,
       629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
       642, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
       673, 674, 676, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705,
       706, 725, 726, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
       738, 760, 762, 763, 764, 765, 766, 767, 768, 769, 771, 794, 795,
       796, 797, 798, 799, 800, 801, 802, 803, 827, 828, 829, 830, 831,
       832, 833, 834, 835, 858, 860, 861, 862, 863, 864, 865, 866, 867,
       868, 892, 893, 894, 895, 896, 897, 898, 900, 925, 926, 927, 928,
       929, 930, 934, 956, 958, 959, 960, 961, 962, 963, 964, 991, 992,
       993, 994, 995])

old_easy = np.array([   0,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
         12,   13,   14,   15,   16,   17,   18,   19,   21,   22,   23,
         24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   35,
         37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,
         48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
         59,   60,   61,   62,   63,   67,   68,   69,   70,   71,   72,
         73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,
         84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,
         95,   96,  101,  102,  103,  104,  105,  106,  107,  108,  109,
        110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,
        121,  122,  123,  124,  125,  126,  127,  133,  134,  135,  136,
        137,  138,  139,  140,  141,  142,  143,  144,  145,  146,  147,
        148,  149,  150,  151,  152,  153,  154,  155,  156,  157,  158,
        159,  165,  166,  167,  168,  169,  170,  171,  172,  173,  174,
        175,  176,  177,  178,  179,  180,  181,  182,  183,  184,  185,
        186,  187,  188,  189,  190,  191,  198,  199,  201,  202,  203,
        204,  205,  206,  207,  208,  209,  210,  211,  212,  213,  214,
        215,  216,  217,  218,  219,  220,  221,  222,  223,  234,  235,
        236,  237,  238,  239,  241,  242,  243,  244,  245,  246,  247,
        248,  249,  250,  251,  252,  253,  254,  255,  262,  264,  266,
        267,  268,  269,  270,  271,  274,  276,  277,  278,  279,  280,
        281,  282,  283,  284,  285,  286,  287,  297,  298,  299,  300,
        301,  302,  304,  305,  310,  314,  315,  316,  317,  318,  319,
        332,  333,  334,  343,  349,  363,  364,  365,  367,  396,  397,
        398,  400,  429,  430,  431,  446,  463,  466,  494,  529,  560,
        593,  594,  659,  660,  691,  692,  693,  694,  695,  759,  761,
        791,  792,  793,  822,  823,  825,  826,  859,  887,  890,  922,
        924,  954,  957,  982,  986,  988,  989,  990, 1015, 1017, 1021,
       1022])

old_normal = np.array([   1,   20,   65,   97,   98,  100,  128,  129,  130,  131,  161,
        162,  163,  164,  193,  194,  195,  196,  197,  227,  228,  229,
        230,  231,  259,  260,  261,  291,  292,  293,  294,  295,  296,
        323,  324,  325,  326,  327,  328,  329,  354,  355,  356,  357,
        358,  359,  360,  361,  362,  387,  388,  389,  390,  391,  392,
        393,  394,  418,  419,  420,  421,  422,  423,  424,  425,  426,
        427,  428,  449,  452,  453,  454,  455,  456,  457,  458,  459,
        460,  483,  484,  485,  486,  487,  488,  489,  490,  491,  492,
        493,  495,  515,  516,  517,  518,  519,  520,  521,  522,  523,
        524,  525,  526,  528,  545,  547,  548,  549,  550,  551,  552,
        553,  554,  555,  556,  557,  558,  559,  580,  581,  582,  583,
        584,  585,  586,  587,  588,  589,  590,  591,  592,  612,  613,
        614,  615,  616,  617,  618,  619,  620,  621,  622,  623,  624,
        626,  643,  644,  645,  646,  647,  648,  649,  650,  651,  652,
        653,  654,  655,  656,  657,  658,  675,  677,  678,  679,  680,
        681,  682,  683,  684,  685,  686,  687,  688,  689,  690,  707,
        708,  709,  710,  711,  712,  713,  714,  715,  716,  717,  718,
        719,  720,  721,  722,  723,  724,  727,  739,  740,  741,  742,
        743,  744,  745,  746,  747,  748,  749,  750,  751,  752,  753,
        754,  755,  756,  757,  758,  770,  772,  773,  774,  775,  776,
        777,  778,  779,  780,  781,  782,  783,  784,  785,  786,  787,
        788,  789,  790,  804,  805,  806,  807,  808,  809,  810,  811,
        812,  813,  814,  815,  816,  817,  818,  819,  820,  821,  824,
        836,  837,  838,  839,  840,  841,  842,  843,  844,  845,  846,
        847,  848,  849,  850,  851,  852,  853,  854,  855,  856,  857,
        869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,
        880,  881,  882,  883,  884,  885,  886,  888,  889,  891,  899,
        901,  902,  903,  904,  905,  906,  907,  908,  909,  910,  911,
        912,  913,  914,  915,  916,  917,  918,  919,  920,  921,  923,
        931,  932,  933,  935,  936,  937,  938,  939,  940,  941,  942,
        943,  944,  945,  946,  947,  948,  949,  950,  951,  952,  953,
        955,  965,  966,  967,  968,  969,  970,  971,  972,  973,  974,
        975,  976,  977,  978,  979,  980,  981,  983,  984,  985,  987,
        996,  997,  998,  999, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
       1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1016, 1018, 1019,
       1020, 1023])

aaa = set(old_hard) - set(new_st.index.to_numpy())

len(aaa)

list(aaa)

bbb = df.loc[list(aaa)]

bbb.loc[bbb['hard_ratio'] == bbb[['easy_ratio', 'normal_ratio', 'hard_ratio', 'new_ratio']].max(axis=1)]


# 특정 구간 그래프 모양 살피기
for a,b in zip(bbb['alpha'][30:70], bbb['beta'][30:70]):
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

easy_st = df.loc[df['easy_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio','new_ratio']].max(axis=1)]

normal_st = df.loc[df['normal_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio','new_ratio']].max(axis=1)]

hard_st = df.loc[df['hard_ratio'] == df[['easy_ratio', 'normal_ratio', 'hard_ratio','new_ratio']].max(axis=1)]

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

