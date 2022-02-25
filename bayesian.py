from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np

x, y = make_classification(n_samples=100, n_features=10, n_classes=2)

rf = RandomForestClassifier()
# 传统的机器学习方法中 交叉验证常常返回均值 可以用来评估模型的好坏 进而选择最佳模型
# 在实际应用中 可与贝叶斯优化器结合使用来调参
print(np.mean(cross_val_score(rf, x, y, cv=5, scoring='roc_auc')))


def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(max_features),
            max_depth=int(max_depth),
            random_state=2,
        ),
        x, y, scoring='roc_auc', cv=5
    ).mean()
    return val


optimizer = BayesianOptimization(
    rf_cv,
    {
        'n_estimators': (10, 200),
        'min_samples_split': (2, 20),
        'max_features': (1, 10),
        'max_depth': (5, 50)
    }
)

optimizer.maximize(init_points=20, n_iter=30)
print(optimizer.max)
# for i, res in enumerate(optimizer.res):
#     # \t可以保证每次迭代的输出在一行
#     print("Iteration {}: \n\t{}".format(i, res))
