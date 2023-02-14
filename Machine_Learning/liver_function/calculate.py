# 提供实测数据的偏离分析 和 预测数据的最佳维持、最长维持时间分析
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def optimise():
    csv = open('liver_data_complement/scaffold-albumin-com-60.csv', encoding='utf-8')
    data = pd.read_csv(csv)

    # 对每一个不重复的Feature元素创建字典 用来保存每个Feature具体提供数据的天数 起始计数为0
    num_dict = dict.fromkeys(data['Feature'], 0)

    statistic = pd.DataFrame(data=None, columns=['Feature', 'MaMiR', 'MaFR', 'MiFR', 'LFFR', 'MFFR',
                                                 'Duration', 'Last'])
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
                fun = data[data['Feature'] == prior]['Albumin'].tolist()
                max_ = np.max(fun)
                min_ = np.min(fun)
                mean_ = np.mean(fun)
                # 计算五个统计值
                MaMiR = np.around(max_ / min_, 2)
                MaFR = np.around((max_ - mean_) / mean_, 2)
                MiFR = np.around((mean_ - min_) / mean_, 2)
                LFFR = np.around((fun[len(fun) - 1] - fun[0]) / fun[0], 2)
                MFFR = np.around((mean_ - fun[0]) / fun[0], 2)
                # 此处开始计算最佳维持时间 最长维持时间
                start = 0
                end = 0
                last = 0
                count_start = 0  # 记录增长点
                count_end = 0  # 记录下降点
                for index_1, value in enumerate(fun):
                    # 根据不同模型的MFFR而改变
                    if index_1 == 0:
                        continue
                    if fun[index_1] >= fun[0] * 1.73 and count_start < 1:
                        start = index_1 + 1
                        count_start += 1
                    elif fun[index_1] < fun[0] * 1.73 and index_1 > start > 1 > count_end:  # 在大于start的索引中寻找end
                        end = index_1 + 1
                        count_end += 1
                    if index_1 == len(fun)-1 and end == 0:
                        end = len(fun)-1
                for index_2, value in enumerate(fun):
                    if fun[index_2] < fun[0] and index_2 > end:
                        last = index_2 + 1
                        break
                # 若一直下降或未达到均值特定倍数 则认为最佳功能维持时间不存在 记为 0
                if start == 0:
                    statistic = statistic.append({'Feature': prior, 'MaMiR': MaMiR, 'MaFR': MaFR, 'MiFR': MiFR,
                                       'LFFR': LFFR, 'MFFR': MFFR, 'Duration': start, 'Last': last}, ignore_index=True)
                else:
                    statistic = statistic.append({'Feature': prior, 'MaMiR': MaMiR, 'MaFR': MaFR, 'MiFR': MiFR,
                                                  'LFFR': LFFR, 'MFFR': MFFR,
                                                  'Duration': '(' + str(start) + ', ' + str(end) + ')',
                                                  'Last': last}, ignore_index=True)
                # prior = row['Feature']  # 更改prior对应的Feature值
    statistic.to_csv('./statistics/scaffold-albumin-60-duration.csv', index=False)


def actual():
    csv = open('liver_data/spheroid-albumin.csv', encoding='utf-8')
    data = pd.read_csv(csv)
    # 对每一个不重复的Feature元素创建字典 用来保存每个Feature具体提供数据的天数 起始计数为0
    num_dict = dict.fromkeys(data['Feature'], 0)

    # 去除预测数据行 即功能分泌指标为空的行
    for index in range(0, len(data)):
        if pd.isnull(data.loc[index, 'Albumin']):
            data.drop(index=index, inplace=True)

    statistic = pd.DataFrame(data=None, columns=['Feature', 'MaMiR', 'MaFR', 'MiFR', 'LFFR', 'MFFR'])
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
                fun = data[data['Feature'] == prior]['Albumin'].tolist()
                max_ = np.max(fun)
                min_ = np.min(fun)
                mean_ = np.mean(fun)
                # 计算五个统计值
                MaMiR = np.around(max_ / min_, 2)
                MaFR = np.around((max_ - mean_) / mean_, 2)
                MiFR = np.around((mean_ - min_) / mean_, 2)
                LFFR = np.around((fun[len(fun) - 1] - fun[0]) / fun[0], 2)
                MFFR = np.around((mean_ - fun[0]) / fun[0], 2)
                statistic = statistic.append({'Feature': prior, 'MaMiR': MaMiR, 'MaFR': MaFR, 'MiFR': MiFR,
                                              'LFFR': LFFR, 'MFFR': MFFR}, ignore_index=True)
    # 下列操作统计每列的最大值 最小值 以及均值的范围
    max_df = statistic.max().tolist()
    max_df = list(map(str, max_df))
    min_df = statistic.min().tolist()
    min_df = list(map(str, min_df))
    mean_df = np.around(statistic.mean().tolist(), 2)
    mean_df = list(map(str, mean_df))
    statistic = statistic.append({'Feature': 'RANGE', 'MaMiR': '('+min_df[1]+', '+max_df[1]+')',
                                  'MaFR': '('+min_df[2]+', '+max_df[2]+')', 'MiFR': '('+min_df[3]+', '+max_df[3]+')',
                                  'LFFR': '('+min_df[4]+', '+max_df[4]+')', 'MFFR': '('+min_df[5]+', '+max_df[5]+')'},
                                 ignore_index=True)
    statistic = statistic.append({'Feature': 'AVERAGE', 'MaMiR': mean_df[0], 'MaFR': mean_df[1],
                                  'MiFR': mean_df[2], 'LFFR': mean_df[3], 'MFFR': mean_df[4]}, ignore_index=True)

    # 将dataframe保存为csv文件
    statistic.to_csv('./statistics/spheroid-albumin.csv', index=False)


# actual()  # 分析实测数据的偏离比例
optimise()  # 分析预测数据的最佳维持时间和最长维持时间


