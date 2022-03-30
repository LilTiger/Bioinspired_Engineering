# 使用说明：可先使用normalization.py将数据正则化为parameters_norm.csv文件【线性回归不需要】
# 根据变量数 更改append到x_label和y_label的索引
# 注意：LogisticRegression()用来分类 LinearRegression()用来回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import f_classif
import csv
from sklearn import preprocessing

csv_file = open('parameters.csv', 'r')
reader = csv.reader(csv_file)

# 从csv文件中读取参数和标签 声明两个空列表先
# 读取到的parameters.csv为 正则化后的数据形式：第一列为索引 之后的列为 自变量-数据 自成一列
x_label = []
y_label = []
for item in reader:
    if reader.line_num == 1:
        continue
    # 每一次迭代 item列表读出csv中的 一行数据
    # item[1:7]表示切割列表元素item[1]至item[7] 但不包含item[7]
    x_label.append(item[1:7])
    y_label.append(item[8])

# clf控制台中的coef_和intercept分别为权重 w b
# 注意 二分类问题中输出的仅为第二类的 w b
clf = LogisticRegression()
x = np.array(x_label)
y = np.array(y_label)
clf.fit(x, y)
# decision_function返回结果的数值表示模型预测样本属于某个类别的可信度
# 在二分类中表示positive正样本的可信度：大于0表示正样本的可信度大于负样本，否则可信度小于负样本
# 在二分类中 第二类（即值更大的类别）为positive正样本

# ys = [[-0.912870929, -0.471404521, -0.683130051, -0.471404521, -0.525785614, -0.702487077]]
ys = [[0, 0, 0, 0, 1, 12]]
# print(clf.decision_function(ys))
print(clf.predict_proba(ys))
print(clf.predict(ys))


# # 以下开始相关性计算(排除第一列 ref的影响 ref的作用仅仅是对比文献提取数据的准确性)
# csv_data = pd.read_csv("parameters.csv")
# # 若是计算parameters_norm.csv 注释掉下面两行即可
# csv_data.drop(['ref'], axis=1, inplace=True)
# csv_data.dropna(axis=1, inplace=True)
# # 计算每个自变量的 相关系数
# corr = csv_data.corr()
# corr.to_csv('correlation.csv')
# # 计算每个自变量的 f_score p_score 前者越大/后者越小 那么与因变量的关系越大 即该自变量越重要
# f_score, p_score = f_classif(x_label, y_label)
# print('f score: ', f_score)
# print('p_score: ', p_score)
# # 以下操作将f_score值转化为百分比 即可近似 特征（自变量）的【贡献度】
# # 保存原数组sum 以免index的迭代中 sum会不断更新 造成不准确
# sums = f_score.sum()
# print(f'contribution rates of {len(x_label[0])} features are:')
# parameters = ['Culture serum', 'Stretching direction', 'Cylic or static stretching', 'Continuous or intermittent',
#               'Amplitude', 'Effective stretching duration']
# for index in range(0, len(f_score)):
#     f_score[index] /= sums
#     # 此种百分比控制输出需掌握
#     print(f'{parameters[index]}: {100*f_score[index]:.2f}% ')

