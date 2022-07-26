# 使用xgboost和f_class(f_regression)计算变量的贡献度【特征重要性】
# 注意数据缺失值fillna的处理方式 对于xgboost来说 也可以对缺失值不做处理
# 数据归一化可在数据预处理阶段直接归一 无需单独开一个文件normalization 且归一化只需对自变量进行
# 注意：LogisticRegression()用来分类 LinearRegression()用来回归
# 特别注意：f_classif 和 f_regression 只能用来寻找变量之间的线性相关关系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, chi2, f_regression
import csv
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler

csv_file = open('evaluation.csv', encoding='ISO-8859-1')
data = pd.read_csv(csv_file)

# 指明csv文件中标签和特征列名即可
# 计算变量重要性时不需要取天数这一变量~
label_name = 'Albumin'
feature_name = ['Day', 'Cell Type', 'Cell Seeding', 'Scaffold Type', 'Modification', 'Concentration', 'Pore Size', 'Thick',
                'Diameter', 'Porosity', 'Static/dynamic']
x_label = data[feature_name]
y_label = data[label_name]

# 对缺失数据 x_label 而不是整个data[evaluation表格中包含了很多中文数据] 进行插值处理
# 无论是否进行 主动插值 贡献率不变
# fillna中 pad为利用前面的数据填充 df.mode()/median()/mean()为众数、中位数、平均值填充
# x_label = x_label.interpolate(method='pad')
# x_label = x_label.fillna(x_label.median())
# 数据归一化处理
scaler = StandardScaler()
columns = x_label.columns
x_label_norm = pd.DataFrame(scaler.fit_transform(x_label), columns=columns)

# 此处可更换具体的拟合模型 xgb库可直接绘制feature_importance图像
# 通过控制n_estimator来控制F_score的范围
clf = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, reg_alpha=0.005, subsample=0.8,
                       gamma=0, colsample_bylevel=0.8)

# 习惯将数据转化为 float 格式
x = np.array(x_label).astype(np.float64)
y = np.array(y_label).astype(np.float64)
# 若要使输出的importance保留标签 注意x_label和y_label需要是dataframe的形式
clf.fit(x_label_norm, y_label)
plot_importance(clf)
# 以下语句针对标签过长显示不全的问题
plt.tight_layout()
plt.show()

# decision_function返回结果的数值表示模型预测样本属于某个类别的可信度
# 在二分类中表示positive正样本的可信度：大于0表示正样本的可信度大于负样本，否则可信度小于负样本
# 在二分类中 第二类（即值更大的类别）为positive正样本
# # ys = [[-0.912870929, -0.471404521, -0.683130051, -0.471404521, -0.525785614, -0.702487077]]
# ys = [[0, 0, 0, 0, 1, 12]]
# # print(clf.decision_function(ys))
# print(clf.predict_proba(ys))
# print(clf.predict(ys))


# 以下开始计算 线性 相关性
csv_data = pd.read_csv('evaluation.csv', encoding='ISO-8859-1')
# # 若是计算parameters_norm.csv 注释掉下面两行即可
# csv_data.drop(['ref'], axis=1, inplace=True)
# csv_data.dropna(axis=1, inplace=True)
# 计算每个自变量的 相关系数
corr = csv_data.corr()
# corr.to_csv('correlation.csv')
# 计算每个自变量的 f_score p_score 前者越大/后者越小 那么与因变量的关系越大 即该自变量越重要
f_score, p_score = f_regression(x, y)
print('f score: ', f_score)
print('p_score: ', p_score)
# 以下操作将f_score值转化为百分比 即可近似 特征（自变量）的【贡献度】
# 保存原数组sum 以免index的迭代中 sum会不断更新 造成不准确
sums = f_score.sum()
print(f'contribution rates of {x_label.shape[1]} features are:')

for index in range(0, len(f_score)):
    f_score[index] /= sums
    # 此种百分比控制输出需掌握
    print(f'{feature_name[index]}: {100*f_score[index]:.2f}% ')

