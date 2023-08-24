"""
原始xgboost模型对于channel width的改变会输出相同的albumin
因此尝试先将特征利用PCA降维后 再训练和预测
此处的输入和预测的输出都为 后缀为fillna的文件 （因PCA主成分转换时要求不能为空）
"""

# 函数预测验证部分 可使用随机的参数进行in silico的实验
# 需要预测和验证的数据放在data_val中 且同样要求【所有原始数据】在上方 之后紧跟【预测数据】
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.decomposition import PCA

csv_file = open('liver_data/chip-albumin - fillna.csv', encoding='utf-8')
data = pd.read_csv(csv_file)

# 采用特定 label_name 和 feature_name 时 对应更改名称或退注释即可
label_name = 'Albumin'

feature_name = ['Day', 'Cell', 'Cell Seeding', 'Co-Cell Seeding', 'Material', 'Material-1-Con', 'Material-2-Con',
                'Modification', 'Modi-1-Con', 'Modi-2-Con', 'Self-circulated', 'Multi-organ', 'Medium',
                'Medium-out', 'Medium-in', 'Serum-out', 'Serum-in', 'Shear Stress', 'Channel Width',
                'Physical-sti', 'Flow Rate']  # chip
"""
# copy()方法创建df的深副本df_deep = df.copy([默认]deep=True) 【可以理解为 创建新的DataFrame并赋值 二者不共享内存空间】
# 即df2重新开辟内存空间存放df_deep的数据 df与df_deep所指向数据的地址不一样而仅对应位置元素一样 故其中一个变量名中的元素发生变化，另一个不会随之发生变化
"""
x_label = data[feature_name].copy()
y_label = pd.DataFrame(data[label_name]).copy()

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

# 选择主成分的数量，可以根据实际需求进行调整
n_components = 10

pca = PCA(n_components=n_components)

# 使用PCA进行训练并转换特征
x_label_pca = pca.fit_transform(x_label)
x_label_pred_pca = pca.transform(x_label_pred)

# 数据标准化 xgboost可略过
scaler = StandardScaler()
columns = x_label.columns  # x_label 和 x_pred 的列变量相同
# 转换后的特征仍然需要进行标准化
x_label_norm = pd.DataFrame(scaler.fit_transform(x_label_pca), columns=[f"PC{i+1}" for i in range(n_components)])
x_label_pred_norm = pd.DataFrame(scaler.transform(x_label_pred_pca), columns=[f"PC{i+1}" for i in range(n_components)])

# 此处对原始csv表格分割后且标准化后的数据 分割训练和测试集并交叉验证
# 对划分之后的DataFrame划分训练集和测试集【某行对应的功能指标为空的行为最终数据填充行 也即预测行 而不是测试集】
test_percent = 0.3
# 注意分割训练/测试集时要对norm后的x_label进行操作
x_train, x_test, y_train, y_test = train_test_split(x_label_norm, y_label, test_size=test_percent, random_state=412)

train_data = xgb.DMatrix(x_train, y_train)
# 少量数据拟合时若测试集精度不高 尝试更改params的值 比如eta num_boost_rounds等
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 12,
    # 'min_child_weight': 6,
    # 'subsample': 0.2,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 1600
xgboost = xgb.train(params=params, dtrain=train_data, num_boost_round=num_boost_rounds)


# 以下开始模型准确性验证部分
# 验证决定系数R平方/平均绝对值误差（MAE）/平均绝对百分比误差（MAPE）/均方误差（MSE）/均方根误差（RMSE）
test_data = xgb.DMatrix(x_test)
y_test_pred = xgboost.predict(test_data)
r2 = r2_score(y_test, y_test_pred)
print("R2 score: %.2f" % r2)
mae = mean_absolute_error(y_test, y_test_pred)
print("MAE: %.2f" % mae)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"MAPE: {100*mape:.2f}%")
mse = mean_squared_error(y_test, y_test_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse ** (1 / 2.0)))
# 模型完整存储了训练的参数 保存和读取模型对结果无任何影响
# joblib.dump(xgboost, 'xgb.pkl')  # 保存模型


# 若输入数据不含预测行 也即不含空值的功能指标行 将以下代码注释即可
# 以下对测试后的模型进行天数预测
if r2 > 0.7:

    # 需要做val的数据
    csv = open('liver_data/chip-albumin-vali - fillna.csv', encoding='utf-8')
    data_val = pd.read_csv(csv)

    x_val = data_val[feature_name].copy()
    y_val = pd.DataFrame(data_val[label_name]).copy()

    x_val_pred = x_val.copy()
    y_val_pred = y_val.copy()
    for index in range(0, len(y_val)):
        if pd.isnull(y_val.loc[index, 'Albumin']):
            x_val.drop(index=index, inplace=True)
            y_val.drop(index=index, inplace=True)
        else:
            x_val_pred.drop(index=index, inplace=True)
            y_val_pred.drop(index=index, inplace=True)

    # 与训练模型数据流程类型 做标准化
    # 同样，预测部分也需要对特征进行PCA转换
    x_val_pca = pca.transform(x_val)
    x_val_pred_pca = pca.transform(x_val_pred)

    x_val_norm = pd.DataFrame(scaler.transform(x_val_pca), columns=[f"PC{i + 1}" for i in range(n_components)])
    x_val_pred_norm = pd.DataFrame(scaler.transform(x_val_pred_pca),
                                   columns=[f"PC{i + 1}" for i in range(n_components)])

    pred_data = xgb.DMatrix(x_val_pred_norm)
    pred_list = np.array(xgboost.predict(pred_data))

    # 丢弃x/y_label_pred中的索引 重排索引
    # 解释一下为什么不必重排原始数据的索引 原始数据直接提取自原DataFrame 索引和数据一一对应 后续直接concat自变量和因变量即可
    # 而预测数据中需将预测的值输入到y_label_pred 且预测结果保存在一个索引从0开始的pred_list 为了使y_label_pred和pred_list索引对应 需要重排
    y_val_pred.reset_index(drop=True, inplace=True)
    x_val_pred.reset_index(drop=True, inplace=True)
    # 填入预测的数据值
    for index in range(pred_list.__len__()):
        y_val_pred.loc[index, 'Albumin'] = pred_list[index]

    # 拼接自变量和因变量 形成完整的原始数据（预测数据）行
    raw_data = pd.concat([y_val, x_val], axis=1)
    raw_data.insert(loc=0, column='Source', value='true')  # 备注数据来源于原始值
    pred_data = pd.concat([y_val_pred, x_val_pred], axis=1)
    pred_data.insert(loc=0, column='Source', value='predicted')  # 备注数据来源于预测值
    # 将预测后的数据拼接到原始数据 形成补点后的dataframe
    final = pd.concat([raw_data, pred_data], axis=0)
    """
    加入原始数据中用于区分同一文章不同条数据特异性的Feature列
    注意! 此时需要确保原始数据和预测数据concat之后的布局跟输入的csv形成的data相同
    故输入的csv 需要确保【所有文章的】原始列在上方 之后紧跟【所有文章的】预测列 （注意不是单篇文章的原始列在上预测列在下然后是下一篇文章）
    """
    final.insert(0, 'Feature', data_val['Feature'].values)

    final.reset_index(drop=True, inplace=True)  # 重排原始+预测序列 得到完整的DataFrame
    final.sort_values(by=['Feature'], ascending=True, inplace=True)  # 按照Feature的类型排序
    final['Albumin'] = final['Albumin'].apply(lambda x: round(x, 3))  # lambda可定义函数 此处为对dataframe的某列数值保留两位小数
    final.to_csv('liver_data_complement/chip-val-pca.csv', index=False, encoding='utf_8_sig')


