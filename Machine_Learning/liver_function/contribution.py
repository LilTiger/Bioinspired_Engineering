# **代码未完成（feature列和xgboost的参数未最终确定）**

# 使用xgboost和f_class(f_regression)计算变量的贡献度【特征重要性】
# 注意数据缺失值fillna的处理方式 对于xgboost来说 也可以对缺失值不做处理
# 数据归一化可在数据预处理阶段直接归一 无需单独开一个文件normalization 且归一化只需对自变量进行
# 注意：LogisticRegression()用来分类 LinearRegression()用来回归
# 特别注意：f_classif 和 f_regression 只能用来寻找变量之间的线性相关关系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_selection import f_regression

csv_file = open('evaluation.csv', encoding='ISO-8859-1')
data = pd.read_csv(csv_file)

label_name = 'Albumin'
feature_name = ['Day', 'Cell Type', 'Cell Seeding', 'Scaffold Type', 'Modification', 'Concentration', 'Pore Size', 'Thick',
                'Diameter', 'Porosity', 'Static/dynamic']
# copy()方法创建df的深副本df_deep = df.copy([默认]deep=True) 【可以理解为 创建新的DataFrame并赋值 二者不共享内存空间】
# 即df2重新开辟内存空间存放df_deep的数据 df与df_deep所指向数据的地址不一样而仅对应位置元素一样 故其中一个变量名中的元素发生变化，另一个不会随之发生变化
x_label = data[feature_name].copy()
y_label = pd.DataFrame(data[label_name]).copy()
# 新建预测变量dataframe 用于从原始数据中drop有原始数据的行
x_label_pred = x_label.copy()
y_label_pred = y_label.copy()

# 当原始数据中包含混杂的原始数据（文章给出）和预测数据（模型预测）时，下面的函数用于提取真实数据；否则只保留真实数据
# 遍历dataframe的每一行
for index in range(0, len(y_label)):
    # 判断某行对应的Albumin是否为空 为空则为预测数据
    # .loc为按标签提取 .iloc为按位置索引提取 (第一个参数为行 第二个参数为列) 此处有 data.loc[:, 'Albumin'] = data.iloc[:, 0]
    if pd.isnull(y_label.loc[index, 'Albumin']):
        y_label.drop(index=index, inplace=True)
        # 自变量标签同样drop掉 预测数据行 即可
        x_label.drop(index=index, inplace=True)
    # 提取预测数据行
    else:
        y_label_pred.drop(index=index, inplace=True)
        x_label_pred.drop(index=index, inplace=True)

# fillna中 pad为利用前面的数据填充 df.mode()/median()/mean()为众数、中位数、平均值填充
# x_label = x_label.interpolate(method='pad')
# x_label = x_label.fillna(x_label.median())
# 数据归一化处理
scaler = StandardScaler()
columns = x_label.columns  # x_label 和 x_pred的列变量相同
x_label_norm = pd.DataFrame(scaler.fit_transform(x_label), columns=columns)
x_label_pred_norm = pd.DataFrame(scaler.fit_transform(x_label_pred), columns=columns)


# 此处对原始csv表格分割后且标准化后的数据 分割训练和测试集并交叉验证
# 对划分之后的DataFrame划分训练集和测试集【注某行对应的功能指标为空的行为最终数据填充行 而不是测试集】
test_percent = 0.3
# 注意分割训练/测试集时要对norm后的x_label进行操作
x_train, x_test, y_train, y_test = train_test_split(x_label_norm, y_label, test_size=test_percent, random_state=412)

# 通过控制n_estimator来控制F_score的范围
train_data = xgb.DMatrix(x_train, y_train)
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'lambda': 0.005,
    'gamma': 0.005,
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.7,
    # 'colsample_bytree': 0.7,
}
num_boost_rounds = 50
xgboost = xgb.train(params=params, dtrain=train_data, num_boost_round=num_boost_rounds)
xgb.plot_importance(xgboost)
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
x = np.array(x_label).astype(np.float64)
y = np.array(y_label).astype(np.float64)
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

