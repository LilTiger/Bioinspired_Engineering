import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

csv = open('liver_data/2D-urea.csv', encoding='utf-8')
data = pd.read_csv(csv)

# 对每一个不重复的Feature元素创建字典 用来保存每个Feature具体提供数据的天数 起始计数为0
num_dict = dict.fromkeys(data['Feature'], 0)

statistic = pd.DataFrame(data=None, columns=['Feature', 'max-min_ratio', 'max_per', 'min_per', 'maintain', 'mean_per'])

# 去除预测数据行 即功能分泌指标为空的行
for index in range(0, len(data)):
    if pd.isnull(data.loc[index, 'Urea']):
        data.drop(index=index, inplace=True)

for index, row in data.iterrows():
    if index == 0:
        # temp = row['Feature']
        continue
    prior = data.at[index-1, 'Feature']
    num_dict[prior] += 1  # 计算每个Feature提供数据的天数
    if row['Feature'] == prior and index != len(data)-1:
        continue
    else:
        # 提供数据的天数大于1才可以计算差值 均值等
        if num_dict[prior] > 1:
            fun = data[data['Feature'] == prior]['Urea'].tolist()
            max_ = np.max(fun)
            min_ = np.min(fun)
            mean_ = np.mean(fun)
            # 计算五个统计值
            max_min_ratio = np.around(max_ / min_, 2)
            max_per = np.around((max_ - mean_)/mean_, 2)
            min_per = np.around((mean_ - min_)/mean_, 2)
            maintain = np.around((fun[len(fun)-1] - fun[0])/fun[0], 2)
            mean_per = np.around((mean_ - fun[0])/fun[0], 2)

            statistic = statistic.append({'Feature': prior, 'max-min_ratio': max_min_ratio, 'max_per': max_per, 'min_per': min_per,
                                           'maintain': maintain, 'mean_per': mean_per}, ignore_index=True)
            # prior = row['Feature']  # 更改prior对应的Feature值
# 下列操作统计每列的最大值 最小值 以及均值的范围
max_df = statistic.max().tolist()
max_df = list(map(str, max_df))
min_df = statistic.min().tolist()
min_df = list(map(str, min_df))
mean_df = np.around(statistic.mean().tolist(), 2)
mean_df = list(map(str, mean_df))
statistic = statistic.append({'Feature': 'RANGE', 'max-min_ratio': '('+min_df[1]+', '+max_df[1]+')',
                              'max_per': '('+min_df[2]+', '+max_df[2]+')', 'min_per': '('+min_df[3]+', '+max_df[3]+')',
                                           'maintain': '('+min_df[4]+', '+max_df[4]+')',
                              'mean_per': '('+max_df[5]+', '+min_df[5]+')'}, ignore_index=True)
statistic = statistic.append({'Feature': 'AVERAGE', 'max-min_ratio': mean_df[0], 'max_per': mean_df[1],
                              'min_per': mean_df[2], 'maintain': mean_df[3], 'mean_per': mean_df[4]}, ignore_index=True)

# 将dataframe保存为csv文件
statistic.to_csv('./statistics/2D-urea.csv', index=False)

