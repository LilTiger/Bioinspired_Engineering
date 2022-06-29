# 待数据量上去之后再使用此xgboost模型 注意此模型在使用时可以有缺失数据
# 每次输出的importance不同的原因是：随机划分了测试集和验证集 解决方法是在train_test_split中指定random_state 使每次划分的训练集和测试集相同
# 此模型中的importance同样计算f_score
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance

import joblib
from sklearn.model_selection import TimeSeriesSplit  # 时序分割
from sklearn.preprocessing import StandardScaler  # 归一化

# for time-series cross-validation set 5 folds
# tscv = TimeSeriesSplit(n_splits=5)


def get_DataFromCsv():
    df = pd.read_csv('parameters.csv')

    return df


# 拆分数据集为x，y
def feature_label_split(data):
    # 获取dataFrame的名
    name_list = data.columns.values.tolist()

    label_name = 'Increased (1) or inhibited (0) differentiation'
    feature_name = ['Culture serum(FBS=0,horse =1)', 'Stretching direction(Radial=0,Uniaxial=1)', 'Cyclic or static stretching(Cyclic=0,static=1)', 'Continuous(0) or intermittent(1)', 'Amplitude (%) (3%=0,8%-9%, 10%=1,15%=2,20%=3)', 'Effective stretching duration (h)']

    x = data[feature_name]
    y = data[label_name]

    return x, y


# def feature_datatime(dat):
#     dat['nowTime'] = dat['nowTime'].astype('datetime64')
#     df_dt = pd.DataFrame(columns=('Year', 'Month', 'Day', 'Hour', 'Minute'))
#
#     df_dt['Hour'] = dat['nowTime'].dt.hour
#     df_dt['Year'] = dat['nowTime'].dt.year
#     df_dt['Month'] = dat['nowTime'].dt.month
#     df_dt['Day'] = dat['nowTime'].dt.day
#     df_dt['Minute'] = dat['nowTime'].dt.minute
#
#     return df_dt


def init_train_data(df):
    # df_out = df.loc[df['OilCanStatus'] != 1]  # 剔除掉进油数据
    # 此处处理缺失数据 可采用不同方法
    # df_out = df_out.dropna(axis=0)  # 直接删除缺失数据
    df = df.interpolate()
    df = df.fillna(df.mean())  # 用该列的均值作填充（不适用于分类标签数据）
    train_x, train_y = feature_label_split(df)
    # xgboost不需要变量变为哑变量

    test_percent = 0.3
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=test_percent, random_state=100)

    return x_train, x_test, y_train, y_test


# 使用XGB Regressor Fit建模训练
def model_fit_regressor(x_train, x_test, y_train, y_test):
    model = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, reg_alpha=0.005, subsample=0.8,
                             gamma=0, colsample_bylevel=0.8, objective='reg:squarederror')
    # reg:linear is now deprecated in favor of reg:squarederror.
    # 'booster':'gblinear'设置，Booster.get_score() results in empty
    # 数据归一化处理
    scaler = StandardScaler()
    columns = x_train.columns
    indexs_train = x_train.index
    x_train = pd.DataFrame(scaler.fit_transform(x_train), index=indexs_train, columns=columns)
    indexs_test = x_test.index
    x_test = pd.DataFrame(scaler.transform(x_test), index=indexs_test, columns=columns)

    model.fit(x_train, y_train)

    score = model.score(x_train, y_train)
    print("Training score: ", score)

    # - cross validataion
    scores = cross_val_score(model, x_train, y_train, cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())

    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    ypred = model.predict(x_test)
    mse = mean_squared_error(y_test, ypred)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % (mse ** (1 / 2.0)))

    x_ax = range(len(y_test))
    # plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

    # plotModelResults(model, X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test, plot_intervals=True,
    #                  plot_anomalies=True)

    # joblib.dump(model, 'OilCan.pkl')

    return model


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plotModelResults(model, X_train, X_test, y_train, y_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    """
    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=5,
                             scoring="neg_mean_absolute_error")
        # scoring='accuracy'  accuracy：评价指标是准确度,可以省略使用默认值
        # cv：选择每次测试折数
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 20
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    # plt.savefig("linear.png")


# 训练模型
def model_train():
    # 读取Excel数据
    df0 = get_DataFromCsv()
    # df0 = df0.drop_duplicates(subset=['nowTime', 'OilCanID', 'OilCode', 'LiquidLevel'], keep='first',
    #                           inplace=False)  # 数据去重

    # df0 = pd.concat([df0, feature_datatime(df0)], axis=1)

    x_train, x_test, y_train, y_test = init_train_data(df0)

    model = model_fit_regressor(x_train, x_test, y_train, y_test)

    # model= model_train_reg(x_train,x_test,y_train,y_test)
    # model.save_model('OilCanXGbLinear.model')  # 保存训练模型
    # 显示重要特征
    plot_importance(model)
    plt.show()


# 模型加载与使用
def model_test():
    df0 = pd.read_csv('parameters.csv')
    # df0 = pd.concat([df0, feature_datatime(df0)], axis=1)

    x_train, x_test, y_train, y_test = init_train_data(df0)

    # 读取Model
    model = joblib.load('OilCan.pkl')
    ypred = model.predict(x_test)
    # plot_tree(model)
    # plt.show()
    print(ypred)


if __name__ == '__main__':
    # 训练并分析模型
    model_train()
    # model_test()
