# 提供实测数据的偏离分析 和 预测数据的最佳维持、最长维持时间分析
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def optimise():
    csv = open('liver_data_complement/scaffold-urea-com-60.csv', encoding='utf-8')
    data = pd.read_csv(csv)

    # 对每一个不重复的Feature元素创建字典 用来保存每个Feature具体提供数据的天数 起始计数为0
    num_dict = dict.fromkeys(data['Feature'], 0)

    statistic = pd.DataFrame(data=None, columns=['Feature', 'Provided Data', 'Start', 'End', 'Duration', 'Last'])
    for index, row in data.iterrows():
        if index == 0:
            # temp = row['Feature']
            continue
        prior = data.at[index-1, 'Feature']
        provide = []
        num_dict[prior] += 1  # 计算每个Feature提供数据的天数
        if row['Feature'] == prior and index != len(data)-1:
            continue
        else:
            # 提供数据的天数大于1才可以计算差值 均值等
            if num_dict[prior] > 1:
                fun = data[data['Feature'] == prior]['Urea'].tolist()
                # 此处开始统计数据提供天数
                feature = data[data['Feature'] == prior]
                for in_fe, row_fe in feature.iterrows():
                    if row_fe['Source'] == 'true':
                        provide.append(row_fe['Day'])
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
                    if 56 <= fun[index_1] <= 83.2 and count_start < 1:
                        start = index_1 + 1
                        count_start += 1
                    elif fun[index_1] < fun[0] * 2.62 and index_1 > start > 1 > count_end:  # 在大于start的索引中寻找end
                        end = index_1 + 1
                        count_end += 1
                    # 当在60天内的某一天达到了最佳 而到60天依然是最佳时
                    if index_1 == len(fun)-1 and end == 0 and start > 0:
                        end = len(fun)
                for index_2, value in enumerate(fun):
                    if fun[index_2] < fun[0] and index_2 > end:
                        last = index_2 + 1
                        break
                duration = end - start
                # 若一直下降或未达到均值特定倍数 则认为最佳功能维持时间不存在 记为 0

                statistic = statistic.append({'Feature': prior, 'Provided Data': provide, 'Start': start, 'End': end,
                                              'Duration': duration, 'Last': last}, ignore_index=True)

                # prior = row['Feature']  # 更改prior对应的Feature值
    statistic.to_csv('./statistics/scaffold-urea-60-duration.csv', index=False)


def actual():
    csv = open('liver_data/scaffold-albumin.csv', encoding='utf-8')
    data = pd.read_csv(csv)
    # 对每一个不重复的Feature元素创建字典 用来保存每个Feature具体提供数据的天数 起始计数为0
    num_dict = dict.fromkeys(data['Feature'], 0)

    # 去除预测数据行 即功能分泌指标为空的行
    for index in range(0, len(data)):
        if pd.isnull(data.loc[index, 'Albumin']):
            data.drop(index=index, inplace=True)

    statistic = pd.DataFrame(data=None, columns=['Feature', 'Max', 'Min', 'MaMiR', 'MaFR', 'MiFR', 'LFFR', 'MFFR'])
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
                '''
                计算分泌量的最大值和最小值 若论文中不需要 可以不考虑 Max 和 Min 列
                '''
                max_ = np.max(fun)
                min_ = np.min(fun)
                mean_ = np.mean(fun)
                # 计算五个统计值
                MaMiR = np.around(max_ / min_, 2)
                MaFR = np.around((max_ - mean_) / mean_, 2)
                MiFR = np.around((mean_ - min_) / mean_, 2)
                LFFR = np.around((fun[len(fun) - 1] - fun[0]) / fun[0], 2)
                MFFR = np.around((mean_ - fun[0]) / fun[0], 2)
                statistic = statistic.append({'Feature': prior, 'Max': np.around(max_, 2), 'Min': np.around(min_, 2),
                                              'MaMiR': MaMiR, 'MaFR': MaFR, 'MiFR': MiFR,
                                              'LFFR': LFFR, 'MFFR': MFFR}, ignore_index=True)
    # 下列操作统计每列的最大值 最小值 以及均值的范围
    max_df = statistic.max().tolist()
    max_df = list(map(str, max_df))
    min_df = statistic.min().tolist()
    min_df = list(map(str, min_df))
    mean_df = np.around(statistic.mean().tolist(), 2)

    # 求所有正数MFFR的均值
    MFFR_list = statistic['MFFR'].tolist()
    positive = []
    for num in MFFR_list:
        if num > 0:
            positive.append(num)
    positive_mean = np.around(np.mean(positive), 2)
    statistic = statistic.append({'Feature': 'RANGE', 'Max': '('+min_df[1]+', '+max_df[1]+')',
                                  'Min': '('+min_df[2]+', '+max_df[2]+')', 'MaMiR': '('+min_df[3]+', '+max_df[3]+')',
                                  'MaFR': '('+min_df[4]+', '+max_df[4]+')', 'MiFR': '('+min_df[5]+', '+max_df[5]+')',
                                  'LFFR': '('+min_df[6]+', '+max_df[6]+')', 'MFFR': '('+min_df[7]+', '+max_df[7]+')'},
                                 ignore_index=True)
    statistic = statistic.append({'Feature': 'AVERAGE', 'Max': mean_df[0], 'Min': mean_df[1], 'MaMiR': mean_df[2],
                                  'MaFR': mean_df[3], 'MiFR': mean_df[4], 'LFFR': mean_df[5], 'MFFR': mean_df[6]},
                                   ignore_index=True)
    statistic = statistic.append({'Feature': 'Positive_MFFR',  'MFFR': positive_mean}, ignore_index=True)

    # 将dataframe保存为csv文件
    statistic.to_csv('./statistics/scaffold-albumin-sti.csv', index=False)


# actual()  # 分析实测数据的偏离比例
optimise()  # 分析预测数据的最佳维持时间和最长维持时间


