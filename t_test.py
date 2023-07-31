import numpy as np
import pandas as pd
from scipy import stats
import glob
import matplotlib.pyplot as plt

def read_csv_files_in_folder(folder_path):
    # 폴더 경로에서 모든 CSV 파일의 경로를 가져옴
    csv_files = glob.glob(folder_path + "/*.csv")

    # CSV 파일들을 저장할 빈 리스트
    dataframes = []

    # 각 CSV 파일을 읽어와 데이터프레임으로 변환하여 리스트에 추가
    for file in csv_files:
        df = pd.read_csv(file, )
        dataframes.append(df)

    return dataframes

# 특정 폴더 경로
folder_path_nonupdate = "simple_test/nonupdate"

folder_path_update = "simple_test/update"

def make_df(folder_path):
    # 폴더에 있는 모든 CSV 파일을 읽어와 리스트로 받음
    csv_dataframes = read_csv_files_in_folder(folder_path)

    concated_df = pd.DataFrame()

    for dataframe in csv_dataframes:
        temp_df = dataframe.iloc[-1, :]
        concated_df = pd.concat([concated_df, temp_df], ignore_index=True, axis=1)

    concated_df = concated_df.transpose()

    concated_df['win_rate'] = concated_df['Wins'] / (concated_df['Losses'] + concated_df['Wins'])

    # Shapiro-Wilk 테스트 실행
    stat, p = stats.shapiro(concated_df['win_rate'])

    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # p-value 검정
    alpha = 0.05
    if p > alpha:
        print("Data appears to be normally distributed (fail to reject H0)")
    else:
        print("Data does not appear to be normally distributed (reject H0)")

    return concated_df

nonupdate = make_df(folder_path_nonupdate)
update = make_df(folder_path_update)

nonupdate['win_rate'].astype(float).describe()
update['win_rate'].astype(float).describe()

group1 = np.array(nonupdate['win_rate'].iloc[:26].astype(float))
group2 = np.array(update['win_rate'].astype(float))

group1.mean()
group2.mean()


# 독립 두 표본 t 검정 실행
t_stat, p_value = stats.ttest_rel(group1,group2)

print('t-statistic=%.3f, p-value=%.3f' % (t_stat, p_value))

# p-value 검정
alpha = 0.05
if p_value > alpha:
    print("The means of the two groups do not appear to be significantly different (fail to reject H0)")
else:
    print("The means of the two groups appear to be significantly different (reject H0)")



