import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, f_classif
import csv

csv_file = open('parameters.csv', 'r')
reader = csv.reader(csv_file)

# 从csv文件中读取参数和标签 声明两个空列表先
# 以下方法要求csv文件格式为
# 第一列存放 文献索引 中间几列存放变量 空一列 最后一列存放标签
# 若不为上述形式 请自行修改append到x_label和y_label的 item[index]
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
clf = LogisticRegression(solver='liblinear')
clf.fit(x_label, y_label)
# decision_function返回结果的数值表示模型预测样本属于某个类别的可信度
# 在二分类中表示positive正样本的可信度：大于0表示正样本的可信度大于负样本，否则可信度小于负样本
# 在二分类中 第二类（即值更大的类别）为positive正样本
print(clf.decision_function([[0, 1, 0, 1, 2, 72]]))
print(clf.predict_proba([[0, 1, 0, 1, 2, 72]]))
print(clf.predict([[0, 1, 0, 1, 2, 72]]))


csv_data = pd.read_csv("parameters.csv")
# 相关性计算(排除第一列 ref的影响 ref的作用仅仅是对比文献提取数据的准确性)
# 注意 drop操作之后返回了新的对象 如果没有令新的变量等于drop这一操作
# 使用 inplace 作用是在原有dataframe对象上修改 可直接返回drop后的结构
csv_data.drop(['ref'], axis=1, inplace=True)
csv_data.dropna(axis=1, inplace=True)
# 计算每个自变量的 相关系数
corr = csv_data.corr()
corr.to_csv('correlation.csv')
# 计算每个自变量的 f_score p_score 前者越大/后者越小 那么与因变量的关系越大 即该自变量越重要
# print(chi2(x_label, y_label))
f_score, p_score = f_classif(x_label, y_label)
print('f score: ', f_score)
print('p_score: ', p_score)
# 以下操作将f_score值转化为百分比 即可近似 特征（自变量）的【贡献度】
# 保存原数组sum 以免index的迭代中 sum会不断更新 造成不准确
sums = f_score.sum()
print(f'contribution rates of {len(x_label[0])} features are:')
parameters = ['Culture serum', 'Stretching direction', 'Cylic or static stretching', 'Continuous or intermittent',
              'Amplitude', 'Effective stretching duration']
for index in range(0, len(f_score)):
    f_score[index] /= sums
    # 此种百分比控制输出需掌握
    print(f'{parameters[index]}: {100*f_score[index]:.2f}% ')

