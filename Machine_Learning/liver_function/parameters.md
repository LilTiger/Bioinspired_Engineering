### 提供了四种模型对两种功能指标的回归模型的较优参数

> 以下参数的优化目标为 reg:gamma
#### Scaffold-Albumin
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 12,
    'min_child_weight': 6,
    'subsample': 0.2,
    'colsample_bytree': 0.8,
}
num_boost_rounds = 1200
`
R2 score: 0.86
MAE: 12.52
MAPE: 1382.79%
MSE: 481.16
RMSE: 21.94
#### Scaffold-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0.24,
    'max_depth': 24,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
num_boost_rounds = 3000
`
R2 score: 0.99
MAE: 43.32
MAPE: 1864.41%
MSE: 8243.75
RMSE: 90.80
#### Spheroid-Albumin
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 12,
    # 'min_child_weight': 12,
    # # 'subsample': 0.2,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 1600
`
R2 score: 0.98
MAE: 6.97
MAPE: 105.01%
MSE: 297.78
RMSE: 17.26
#### Spheroid-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0.6,
    'max_depth': 12,
    'min_child_weight': 3,
    # 'subsample': 0.2,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 2000
`
R2 score: 0.77
MAE: 91.41
MAPE: 42.00%
MSE: 19282.77
RMSE: 138.86
#### Chip-Albumin
`
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
`
R2 score: 0.87
MAE: 12.83
MAPE: 34.09%
MSE: 2139.77
RMSE: 46.26
#### Chip-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0.24,
    'max_depth': 24,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
num_boost_rounds = 3000
`
R2 score: 0.74
MAE: 16.02
MAPE: 35.05%
MSE: 1155.86
RMSE: 34.00
#### 2D-Albumin
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0.2,
    'max_depth': 12,
    # 'min_child_weight': 6,
    # 'subsample': 0.2,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 1600
`
R2 score: 0.79
MAE: 7.10
MAPE: 186.29%
MSE: 409.44
RMSE: 20.23
#### 2D-Urea
`
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
`
R2 score: 0.75
MAE: 54.71
MAPE: 101.08%
MSE: 12951.06
RMSE: 113.80


> 其它回归模型对于的性能比较 (scaffold-albumin)
#### KNN (default parameter)
R2 score: 0.58
MAE: 21.74
MAPE: 1413.69%
MSE: 1491.60
RMSE: 38.62

#### Random Forest (n_estimator = 160)
R2 score: 0.82
MAE: 13.31
MAPE: 1920.73%
MSE: 616.36
RMSE: 24.83

#### SGD Regressor (default parameter)
R2 score: 0.16
MAE: 36.46
MAPE: 6696.72%
MSE: 2967.94
RMSE: 54.48

#### MLP Regressor (solver='sgd)
R2 score: 0.74
MAE: 18.75
MAPE: 1995.08%
MSE: 923.01
RMSE: 30.38

