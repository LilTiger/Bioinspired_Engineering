# 机器学习回归/分类时 自变量量纲的不统一 【有可能】会造成 贡献度 的计算偏差
# 此文件将csv中的数据 归一化 并存储到新的文件中
# 使用说明：首先将数据存于parameters.csv 格式随意 之后根据csv文件更改label中的列名（自变量） 执行即可归一化生成parameters_norm.csv
import pandas as pd
import csv
from sklearn import preprocessing

# read_csv方法默认会在开头加入新的unnamed列 设置index_col=0可以避免此现象
pd_data = pd.read_csv('parameters.csv', index_col=0)
norm = []
# 可以将csv中带逗号的名称直接复制过来 此种寻找 名称对应列 的好处是 不会对csv中其它列(如分类标签等)造成影响
# 注意 标签 同样需要归一化
label = ["Culture serum(FBS=0,horse =1)", "Stretching direction(Radial=0,Uniaxial=1)", "Cyclic or static stretching(Cyclic=0,static=1)", "Continuous(0) or intermittent(1)", "Amplitude (%) (3%=0,8%-9%, 10%=1,15%=2,20%=3)", "Effective stretching duration (h)", "Increased (1) or inhibited (0) differentiation"]
for i in label:
    # 根据列名 获取到这一列的数据 // x[:,i]为取所有行的第 i 个数据
    y = pd_data.loc[:, i]
    ys = list(preprocessing.scale(y))  # 归一化
    # norm为嵌套列表 外层列表数为 变量数 内层列表数为 每个变量的数据条目数
    norm.append(ys)

# 首先声明一个空的DataFrame
data = pd.DataFrame()
for i in range(len(label)):
    # 向DataFrame中插入数据 按照 自变量-数值 成列的方式构造
    data.insert(i, label[i], norm[i])
data.to_csv('parameters_norm.csv')

print('完毕')
