# 拟合函数 使用了xgboost原生的拟合函数train()和交叉验证函数cv() 而sklearn封装的XGBRegressor略有不同 不可持续学习【参见xgb_note】
# 适用于csv_file中为 全部原始数据 或 原始数据+预测数据行共存（预测数据行即 补充的 文章未给出的天数对应的Urea为空的行）
# xgboost无须数据标准化
# **为附加知识点
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import joblib
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# 一些绘图函数
# 绘图调参 评价标准metrics='RMSE' 其它标准可见xgb_note
def parameters():
    fig, ax = plt.subplots(1)
    params1 = {
        'eta': 0.1,
        'objective': 'reg:gamma',
        'lambda': 0.005,
        'gamma': 0,
        'max_depth': 8,
        'min_child_weight': 3,
        # 'subsample': 0.7,
        # 'colsample_bytree': 0.7,
    }
    params2 = {
        'eta': 0.1,
        'objective': 'reg:gamma',
        'lambda': 0.005,
        'gamma': 0,
        'max_depth': 8,
        'min_child_weight': 3,
        # 'subsample': 0.7,
        # 'colsample_bytree': 0.7,
    }
    cv_test1 = xgb.cv(params=params1, dtrain=train_data, num_boost_round=num_boost_rounds, folds=k_fold, metrics='rmse')
    cv_test2 = xgb.cv(params=params2, dtrain=train_data, num_boost_round=600, folds=k_fold, metrics='rmse')
    ax.plot(cv_test1.iloc[:, 0], c='red', label='train')
    ax.plot(cv_test1.iloc[:, 2], c='blue', label='test')
    ax.plot(cv_test2.iloc[:, 0], c='orange', label='train-modi')
    ax.plot(cv_test2.iloc[:, 2], c='green', label='test-modi')

    ax.legend()
    plt.show()

def plot_2D():
    x = np.linspace(1, 15, 15)

    y = final.iloc[33:37, 1:3]
    xx = final.iloc[272:283, 1:3]
    y = pd.concat([y, xx], axis=0)
    y.sort_values(by=['Day'], inplace=True)
    y.drop('Day', axis=1, inplace=True)
    plt.figure()
    # 以下方法通过折线图在下 散点图（文章提供的真实数据点）在上的方式形成 真实+预测 拟合曲线
    plt.plot(x, y, marker='o', markerfacecolor='white', label='2D', zorder=1)  # 可以通过zorder设置不同曲线的优先级

    plt.scatter([5, 9, 12, 15], [5.5, 0.6, 0.16, 0.16], marker='o', zorder=2)  # 可以通过zorder设置不同曲线的优先级

    plt.title('Urea Secretion Line Chart')
    plt.xlabel('Day')
    plt.ylabel('Urea (μg/day/10^6 cells)')
    plt.legend()
    plt.savefig('pics/2d line chart.svg', dpi=300)  # 用png可以无损压缩 jpg有损压缩 svg效果更好
    plt.show()

# 将原始和预测数据绘制折线图 此处需根据不同文章的Feature作更改
def plot():
    x = np.linspace(1, 15, 15)
    y1 = final.loc[final['Feature'] == '13-0']['Urea']
    y2 = final.loc[final['Feature'] == '13-100']['Urea']
    y3 = final.loc[final['Feature'] == '13-25']['Urea']
    y4 = final.loc[final['Feature'] == '13-50']['Urea']
    y5 = final.loc[final['Feature'] == '13-75']['Urea']

    plt.figure()
    # 以下方法通过折线图在下 散点图（文章提供的真实数据点）在上的方式形成 真实+预测 拟合曲线
    plt.plot(x, y1, marker='D', markerfacecolor='white', label='Flow: 0', zorder=1)  # 可以通过zorder设置不同曲线的优先级
    plt.plot(x, y2, marker='D', markerfacecolor='white', label='Flow: 100', zorder=1)
    plt.plot(x, y3, marker='D', markerfacecolor='white', label='Flow: 25', zorder=1)  # 可以通过zorder设置不同曲线的优先级
    plt.plot(x, y4, marker='D', markerfacecolor='white', label='Flow: 50', zorder=1)
    plt.plot(x, y5, marker='D', markerfacecolor='white', label='Flow: 75', zorder=1)  # 可以通过zorder设置不同曲线的优先级
    # 记得先画 y1 再画 y2 以此类推
    plt.scatter([1, 5, 10, 15], [26.44, 25.20, 20.34, 17.63], marker='D', zorder=2)  # 可以通过zorder设置不同曲线的优先级
    plt.scatter([1, 5, 10, 15], [34.35, 28.02, 25.76, 20.34], marker='D', zorder=2)  # 可以通过zorder设置不同曲线的优先级
    plt.scatter([1, 5, 10, 15], [30.39, 28.93, 25.65, 22.82], marker='D', zorder=2)  # 可以通过zorder设置不同曲线的优先级
    plt.scatter([1, 5, 10, 15], [35.71, 34.24, 33.56, 31.75], marker='D', zorder=2)  # 可以通过zorder设置不同曲线的优先级
    plt.scatter([1, 5, 10, 15], [34.35, 30.85, 26.67, 24.97], marker='D', zorder=2)  # 可以通过zorder设置不同曲线的优先级


    plt.title('Urea Secretion Line Chart')
    plt.xlabel('Day')
    plt.ylabel('Urea (μg/day/10^6 cells)')
    plt.legend()
    plt.xticks(list(range(1, 16, 2)))
    plt.savefig('pics/chip line chart.svg', dpi=300)  # 用png可以无损压缩 jpg有损压缩 svg效果更好
    plt.show()


csv_file = open('liver_data/scaffold-urea.csv', encoding='utf-8')
data = pd.read_csv(csv_file)

# 采用特定 label_name 和 feature_name 时 对应更改名称或退注释即可
label_name = 'Urea'
feature_name = ['Day', 'Cell', 'Cell Seeding', 'Co-Cell Seeding', 'Co-Cell Seeding-2', 'Scaffold', 'Scaffold-1-Con',
                'Scaffold-2-Con', 'Scaffold-3-Con', 'Modification', 'Modi-1-Con', 'Modi-2-Con', 'Pore Size', 'Diameter',
                'Thick', 'Porosity', 'Flow Rate', 'Fabrication', 'Fabr-para1', 'Fabr-para2', 'Fabr-para3']  # scaffold
# feature_name = ['Day', 'Cell', 'Cell Seeding', 'Co-Cell Seeding', 'Spheroid-Dia', 'Tethered', 'Tethered Film',
#                 'Modification', 'Flow Rate']  # spheroid
# feature_name = ['Day', 'Cell', 'Cell Seeding', 'Co-Cell Seeding', 'Material', 'Material-1-Con', 'Material-2-Con',
#                 'Modification', 'Modi-1-Con', 'Modi-2-Con', 'Self-circulated', 'Multi-organ', 'Medium',
#                 'Medium-out', 'Medium-in', 'Serum-out', 'Serum-in', 'Shear Stress', 'Channel Width',
#                 'Physical-sti', 'Flow Rate']  # chip
# feature_name = ['Day', 'Cell', 'Cell Seeding', 'Coat', 'Co-Cell Seeding']  # 2D
"""
# copy()方法创建df的深副本df_deep = df.copy([默认]deep=True) 【可以理解为 创建新的DataFrame并赋值 二者不共享内存空间】
# 即df2重新开辟内存空间存放df_deep的数据 df与df_deep所指向数据的地址不一样而仅对应位置元素一样 故其中一个变量名中的元素发生变化，另一个不会随之发生变化
"""
x_label = data[feature_name].copy()
y_label = pd.DataFrame(data[label_name]).copy()

# """
# 不能比较大小的属性，通常我们不能用简单的数值来粗暴替换。因为属性的数值大小会影响到权重矩阵的计算，不存在大小关系的属性，其权重也不应该发生相应的变化
# """
# # 对枚举型变量采用编码方式One-Hot
# onehot = OneHotEncoder()
# onehot_feature = pd.DataFrame(onehot.fit_transform(x_label[['Cell', 'Scaffold']]).toarray())
# x_label.drop(columns=['Cell', 'Scaffold'], inplace=True)
# x_label_norm = pd.concat([x_label, onehot_feature], axis=1)
# 新建预测数据结构dataframe 用于从原始数据中drop有原始数据的行
x_label_pred = x_label.copy()
y_label_pred = y_label.copy()

# 当原始数据中包含混杂的原始数据（文章给出）和预测数据（模型预测）时，下面的函数用于提取真实数据；否则只保留真实数据
# 遍历dataframe的每一行
for index in range(0, len(y_label)):
    # 判断某行对应的Urea是否为空 为空则为预测数据
    # .loc为按标签提取 .iloc为按位置索引提取 (第一个参数为行 第二个参数为列) 此处有 data.loc[:, 'Urea'] = data.iloc[:, 0]
    if pd.isnull(y_label.loc[index, 'Urea']):
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
# 数据标准化 xgboost可略过
scaler = StandardScaler()
columns = x_label.columns  # x_label 和 x_pred 的列变量相同
x_label_norm = pd.DataFrame(scaler.fit_transform(x_label), columns=columns)
x_label_pred_norm = pd.DataFrame(scaler.fit_transform(x_label_pred), columns=columns)

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
    'gamma': 0.1,
    'max_depth': 24,
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 3000
xgboost = xgb.train(params=params, dtrain=train_data, num_boost_round=num_boost_rounds)

# """
# 以下开始计算贡献度
# """
# # 对于分类变量，由于天生能用于分割的点就比较少，很容易被"weight"指标所忽略；故使用gain最可以代表特征的重要性
# xgb.plot_importance(xgboost, importance_type='gain', xlabel='gain', max_num_features=10, show_values=False)  # 论文中可设置show_values为False
# plt.tight_layout()
# # 调整左边距以解决features显示不全的问题
# plt.gcf().subplots_adjust(left=0.22)
# # plt.savefig('pics/scaffold-urea.svg', dpi=300)
# plt.show()

# """
# 画出树结构
# """
# tree = xgb.to_graphviz(xgboost, num_trees=2)
# tree.view("trees/tree")

# 首先对于训练集的k折交叉验证 评估模型的rmse【只需要训练集】
k_fold = KFold(n_splits=5, shuffle=True, random_state=412)
# 不作为输出只作为调参 比如设定不同的rounds 检测均方根误差rmse从多少rounds开始趋于稳定
cv_score = xgb.cv(params=params, dtrain=train_data, num_boost_round=num_boost_rounds, folds=k_fold, metrics='rmse')
# **附加知识点** 利用cross-validation score验证该数据集下验证不同模型的准确率【模型选择】
# knn = KNeighborsRegressor()
# knn.fit(x_train, y_train)
# print("K-Nearest Neighbors cross-validation score:", cross_val_score(knn, x_train, y_train, scoring='explained_variance',
#                                                                      cv=k_fold).mean())  # 注意此处scoring可指定评价的具体指标~


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
# 输出模型测试集的实际值和预测值
# print('The true label of test set is:', np.array(y_test).T)
# print('The predicted output of test set is:', y_test_pred)
joblib.dump(xgboost, 'xgb.pkl')  # 保存模型


# 若输入数据不含预测行 也即不含空值的功能指标行 将以下代码注释即可
# 以下对测试后的模型进行天数预测
if r2 > 0.8:
    xgboost = joblib.load('xgb.pkl')
    # 当有新数据需要增量学习时 使用以下指令【注意eta学习率需要酌情变小 类似于微调】
    # model = xgb.train(params=params, dtrain=test_data, num_boost_round=num_boost_rounds, xgb_model=xgboost)
    pred_data = xgb.DMatrix(x_label_pred_norm)
    pred_list = np.array(xgboost.predict(pred_data))
    # pred_list = [round(i, 2) for i in pred_list]  # 一行语句包含对整个list的循环保留特定小数位数操作
    # print('The predicted output of complementary data is:', pred_list)  # 验证模型实际预测能力需要

    # 丢弃x/y_label_pred中的索引 重排索引
    # 解释一下为什么不必重排原始数据的索引 原始数据直接提取自原DataFrame 索引和数据一一对应 后续直接concat自变量和因变量即可
    # 而预测数据中需将预测的值输入到y_label_pred 且预测结果保存在一个索引从0开始的pred_list 为了使y_label_pred和pred_list索引对应 需要重排
    y_label_pred.reset_index(drop=True, inplace=True)
    x_label_pred.reset_index(drop=True, inplace=True)
    # 填入预测的数据值
    for index in range(pred_list.__len__()):
        y_label_pred.loc[index, 'Urea'] = pred_list[index]

    # 拼接自变量和因变量 形成完整的原始数据（预测数据）行
    raw_data = pd.concat([y_label, x_label], axis=1)
    raw_data.insert(loc=0, column='Source', value='true')  # 备注数据来源于原始值
    pred_data = pd.concat([y_label_pred, x_label_pred], axis=1)
    pred_data.insert(loc=0, column='Source', value='predicted')  # 备注数据来源于预测值
    # 将预测后的数据拼接到原始数据 形成补点后的dataframe
    final = pd.concat([raw_data, pred_data], axis=0)
    """
    加入原始数据中用于区分同一文章不同条数据特异性的Feature列
    注意! 此时需要确保原始数据和预测数据concat之后的布局跟输入的csv形成的data相同
    故输入的csv 需要确保【所有文章的】原始列在上方 之后紧跟【所有文章的】预测列 （注意不是单篇文章的原始列在上预测列在下然后是下一篇文章）
    """
    final.insert(0, 'Feature', data['Feature'].values)

    final.reset_index(drop=True, inplace=True)  # 重排原始+预测序列 得到完整的DataFrame
    final.sort_values(by=['Feature', 'Day'], ascending=True, inplace=True)  # 先按照Feature的类型排序 在Feature内部再按照Day升序排列
    final['Urea'] = final['Urea'].apply(lambda x: round(x, 2))  # lambda可定义函数 此处为对dataframe的某列数值保留两位小数
    # final.to_csv('final.csv', index=False)

    # plot()  # 将原始点和预测点绘制折线图

