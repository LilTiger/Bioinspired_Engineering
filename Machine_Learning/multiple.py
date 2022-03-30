import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import csv

csv_file = open('parameters.csv', 'r')
reader = csv.reader(csv_file)

# 从csv文件中读取参数和标签 声明两个空列表先
# 读取到的parameters.csv为 正则化后的数据形式：第一列为索引 之后的列为 自变量-数据 因变量-数据
x_label = []
y_label = []
for item in reader:
    if reader.line_num == 1:
        continue
    x_label.append(item[1:7])
    y_label.append(item[8:10])

clf = MultiOutputRegressor(GradientBoostingRegressor())
# 数据类型转换为numeric
x = np.array(x_label).astype(np.float64)
y = np.array(y_label).astype(np.float64)
clf.fit(x, y)

if __name__ == '__main__':
    pre = [[0., 1., 0., 1., 2., 72.]]
    # # 以下为标准化操作 注意若是一维数组（一行自变量数据预测） 则不能完成归一化
    # # 标准化是对列操作的,一维数组每列中只有一个值,无法计算.会导致输出的均值化结果为全0.可通过reshape(-1, 1)将一维数组改为二维数组.
    # scalar = StandardScaler()
    # pre = scalar.fit_transform(pre)
    # # 可以使用inverse_transform将标准化后的数组 还原
    # pre_ = scalar.inverse_transform(pre)

    result = clf.predict(pre)
    # result = np.around(result, 3)
    print(result)
    print(clf.score(x, y))

