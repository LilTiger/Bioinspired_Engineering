# 贝叶斯优化器 可根据需要寻找最优参数
# 可适用于多输出回归/分类任务的优化
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np
import csv

# 根据csv数据 寻找最优参数
csv_file = open('parameters.csv', 'r')
reader = csv.reader(csv_file)
x_label = []
y_label = []
for item in reader:
    if reader.line_num == 1:
        continue
    x_label.append(item[1:7])
    y_label.append(item[8:10])
x = np.array(x_label).astype(np.float64)
y = np.array(y_label).astype(np.float64)

# x, y = make_regression(n_samples=1000, n_features=10, random_state=1)

# 此处跟贝叶斯优化器中的回归器无关 下面语句的作用只是 提前输出一个交叉验证的评估值
rf = MultiOutputRegressor(GradientBoostingRegressor())
rf.fit(x, y)
score = rf.score(x, y)
print(np.mean(score))


# 此处开始定义贝叶斯优化器的核函数
def rf_cv(n_estimators, learning_rate, min_samples_split, max_depth, max_features):
    # 此处定义 分类/回归器 的参数
    clf = MultiOutputRegressor(GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_depth=int(max_depth),
        max_features=int(max_features)
    ))
    clf.fit(x, y)
    val = clf.score(x, y)
    return val


optimizer = BayesianOptimization(
    rf_cv,
    pbounds=
    {
        'n_estimators': (10, 200),
        'learning_rate': (0.0001, 0.1),
        'min_samples_split': (2, 10),
        'max_depth': (3, 20),
        'max_features': (1, 6),
    }
)

optimizer.maximize(init_points=50, n_iter=100)
print(optimizer.max)
# for i, res in enumerate(optimizer.res):
#     # \t可以保证每次迭代的输出在一行
#     print("Iteration {}: \n\t{}".format(i, res))
