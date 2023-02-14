# 扩充预测列
import pandas as pd
import xgboost as xgb
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

csv = open('liver_data/chip-urea.csv', encoding='utf-8')
data = pd.read_csv(csv)

# 抛弃文件中补点的预测列 直接重新扩充
for index in range(0, len(data)):
    if pd.isnull(data.loc[index, 'Urea']):
        data.drop(index=index, inplace=True)

feature_array = data['Feature'].unique()
for element in feature_array:
    feature = data.loc[data['Feature'] == element]
    day = data.loc[data['Feature'] == element]['Day'].tolist()
    # 若要预测其它对应的天数 在此处更改
    for index in range(1, 61):
        # 当已有的实测点未包含该index时 补全该预测列
        if index not in day:
            data = data.append(feature.iloc[0], ignore_index=True)
            data.loc[len(data)-1, 'Day'] = index
            data.loc[len(data)-1, 'Urea'] = np.nan

data.to_csv('liver_data/chip-urea-60.csv', index=False)

