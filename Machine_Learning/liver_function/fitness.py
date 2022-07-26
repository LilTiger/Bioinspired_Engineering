# 拟合函数
# 适用于csv_file中为 全部原始数据 或 原始数据+预测数据行共存（预测数据行即 补充的 文章未给出的天数对应的Albumin为空的行）
# **为附加知识点
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
import joblib
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


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
xgboost = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, reg_alpha=0.005, subsample=0.8,
                           gamma=0, colsample_bylevel=0.8)
# # 习惯将数据转化为 float 格式
# x = np.array(x_label).astype(np.float64)
# y = np.array(y_label).astype(np.float64)

xgboost.fit(x_train, y_train)

# 以下开始模型准确性验证部分
# 首先对于训练集的部分
score = xgboost.score(x_train, y_train)
print("Training score:", score)  # 输出训练集的决定系数R平方
# k折交叉交叉验证 评估模型的准确度【只需要训练集】
k_fold = KFold(n_splits=5, shuffle=True, random_state=412)
kf_score = cross_val_score(xgboost, x_train, y_train, cv=k_fold, scoring='explained_variance')  # 注意此处scoring可指定评价的具体指标~
print("K-Fold cross-validation score: %.2f" % kf_score.mean())
# **附加知识点** 利用cross-validation score验证该数据集下验证不同模型的准确率【模型选择】
# knn = KNeighborsRegressor()
# knn.fit(x_train, y_train)
# print("K-Nearest Neighbors cross-validation score:", cross_val_score(knn, x_train, y_train, cv=k_fold).mean())


# 下面是基于测试集的验证部分
# 验证平均绝对值误差（MAE）/平均绝对百分比误差（MAPE）/均方误差（MSE）/均方根误差（RMSE）
y_test_pred = xgboost.predict(x_test)
mae = mean_absolute_error(y_test, y_test_pred)
print("MAE: %.2f" % mae)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"MAPE: {mape:.2f}% ")
mse = mean_squared_error(y_test, y_test_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse ** (1 / 2.0)))

joblib.dump(xgboost, 'xgb.pkl')  # 保存模型


# 以下对测试后的模型进行天数预测
# 注意 最好在测试准确率较高时真正输出预测数据
# if  mape < 0.1:
#     xgboost = joblib.load('xgb.pkl')
#     pred_list = xgboost.predict(x_label_pred_norm)
#     # 一行语句包含对整个list的循环保留特定小数位数操作
#     # pred_list = [round(i, 2) for i in pred_list]
#     print(pred_list)
#
#     # 丢弃x/y_label_pred中的索引 重排索引
#     # 解释一下为什么不必重排原始数据的索引 原始数据直接提取自原DataFrame 索引和数据一一对应 后续直接concat自变量和因变量即可
#     # 而预测数据中需将预测的值输入到y_label_pred 输出结果保存到一个索引从0开始的pred_list 为了使y_label_pred和pred_list索引对应 需要重排
#     y_label_pred.reset_index(drop=True, inplace=True)
#     x_label_pred.reset_index(drop=True, inplace=True)
#     # 填入预测的数据值
#     for index in range(pred_list.__len__()):
#         y_label_pred.loc[index, 'Albumin'] = pred_list[index]
#
#     # 拼接自变量和因变量 形成完整的原始数据（预测数据）行
#     raw_data = pd.concat([y_label, x_label], axis=1)
#     raw_data.insert(loc=0, column='Source', value='true')  # 备注数据来源于原始值
#     pred_data = pd.concat([y_label_pred, x_label_pred], axis=1)
#     pred_data.insert(loc=0, column='Source', value='predicted')  # 备注数据来源于预测值
#     # 将预测后的数据拼接到原始数据 形成补点后的dataframe
#     final = pd.concat([raw_data, pred_data], axis=0)
#
#     final.reset_index(drop=True, inplace=True)  # 重排原始+预测序列 得到完整的DataFrame
#     final.sort_values(by='Day', ascending=True, inplace=True)  # 按照Day排序
#     # final.to_csv('final.csv', index=False)
