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

#### Scaffold-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 8,
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 3000
`

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

#### Spheroid-Urea
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0,
    'max_depth': 12,
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 1200
`

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

#### Chip-Albumin
`
params = {
    'eta': 0.1,
    'objective': 'reg:gamma',
    'alpha': 0.005,
    'gamma': 0.1,
    'max_depth': 12,
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
}
num_boost_rounds = 2000
`

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

#### 2D-Albumin
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