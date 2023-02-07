## XGBoost有两种接口：

`原生接口，比如xgboost.train，xgboost.cv`
`sklearn接口，比如xgboost.XGBClassifier，xgboost.XGBRegressor`
两种接口有些许不同，比如原生接口的学习率参数是eta，sklearn接口的是learning_rate，原生接口要在train或cv函数中传入num_round作为基学习器个数
而sklearn接口在定义模型时使用参数n_estimators。sklearn接口的形式与sklearn中的模型保持统一，方便sklearn用户学习

# 注意GridSearchCV方法只可搭配sklearn接口的xgboost使用
如果要对XGBoost模型进行交叉验证，可以使用原生接口的交叉验证函数xgboost.cv
对于sklearn接口，可以使用sklearn.model_selection中的cross_val_score，cross_validate，validation_curve三个函数

sklearn.model_selection中的三个函数区别：
cross_val_score最简单，返回模型给定参数的验证得分，不能返回训练得分
cross_validate复杂一些，返回模型给定参数的训练得分、验证得分、训练时间和验证时间等，甚至还可以指定多个评价指标
validation_curve返回模型指定一个参数的一系列候选值的训练得分和验证得分，可以通过判断拟合情况来调整该参数，也可以用来画validation_curve

## xgb.cv()中 metrics
error:    binary classification error rate
rmse:     Rooted mean square error
logloss:  negative log-likelihood function
mae:      Mean absolute error
mape:     Mean absolute percentage error
auc:      Area under curve
aucpr:    Area under PR curve
merror:   Exact matching error, used to evaluate multi-class classification

## xgb.train()中 params：objective 参数默认值为 reg:squarederror

reg:squarederror：以均方差（即 MSE）损失函数为最小化的回归问题任务。
reg:squaredlogerror：以均方根对数误差为最小化的回归问题任务。
reg:logistic：逻辑回归的二分类，评估默认使用均方根误差（rmse）。
reg:pseudohubererror：以 Pseudo-Huber 损失函数的回归问题。
reg:gamma：使用对数链接（log-link）进行伽马回归。输出是伽马分布的平均值。例如，对于建模保险索赔严重性或对可能是伽马分布的任何结果，它可能很有用。
reg:tweedie：使用对数链接（log-link）进行 Tweedie 回归。常用于建模保险的总损失，或用于可能是 Tweedie-distributed 的任何结果。
binary:logistic：逻辑回归的二分类，输出的也是分类的概率，和 reg:logistic 一样，不同的是默认采用错误率评估指标。
binary:logitraw：逻辑回归的二分类，但在进行逻辑回归转换之前直接输出分类得分。
binary:hinge：基于 Hinge 损失函数的二分类，预测输出不是 0 就是 1，而不是分类的概率值。
count:poisson：基于泊松回归的计数任务，输出泊松分布的平均值。
max_delta_step：可以设置该值，默认为 0.7。
survival:cox：基于 Cox 风险比例回归模型的生存分析任务，如癌症患者生存概率等。
survival:aft：基于加速失效模型（aft）的生存分析任务。
aft_loss_distribution：概率密度函数，基于 survival:aft 和 aft-nloglik 作为评价指标。
multi:softmax：使用 softmax 多分类器的多分类任务，返回预测的类别，同时也要设置分类的个数。
multi:softprob：和 softmax 一样，但是输出的一个形式为 ndata * nclass 的向量，可以进一步将矩阵 reshape 成 ndata * nclass 的指标，输出的是每个类别的概率值。
rank:pairwise：使用 LambdaMART 进行文档对方法排名（pairwise），并使成对损失最小化。
rank:ndcg：使用 LambdaMART 进行文档列表方法排名（listwise），并使标准化折让累积收益（NDCG）最大化。
rank:map：使用 LambdaMART 进行文档列表方法排名（listwise），并使平均准确率（MAP）最大化。


## 树模型并非是尺度敏感的 标准化与否不会对贡献度和测试准确性造成影响 

