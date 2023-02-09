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
R2 score: 0.97
MAE: 53.20
MAPE: 1824.71%
MSE: 17847.18
RMSE: 133.59
#### Spheroid-Albumin
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
R2 score: 0.97
MAE: 53.20
MAPE: 1824.71%
MSE: 17847.18
RMSE: 133.59
#### Spheroid-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 24,
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 1600
`
R2 score: 0.85
MAE: 146.59
MAPE: 45.55%
MSE: 62627.38
RMSE: 250.25
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