# 切片范围以x,y的形式给出 请仔细观察切片代码上方对应注释内容
import pandas as pd

# 读取数据
data = pd.read_csv('scopus - heart.csv')

# 取作者-机构所有列（形式为data.iloc[:, x:y])
df = data.iloc[:, 12:44]
df.drop([0], inplace=True)  # 去掉首行

# range的值即为df的所有列范围（索引从0开始）【range的值为0, y-x】
for index in range(0, 32):

        split_1 = df.iloc[:, index].astype('str').str.split(', ', 1, expand=True)

        split_1.columns = ['作者机构' + str(index) + '_' + str(i) for i in split_1.columns]
        split_2 = split_1['作者机构' + str(index) + '_' + str(1)].str.rsplit(', ', 1, expand=True)
        split_2.columns = [['作者机构' + str(index) + '_' + str(i+1) for i in split_2.columns]]

        # 加入split_1中分割的作者和split_2中分割的机构和国家
        results_temp = df.join(split_1[['作者机构' + str(index) + '_' + str(0)]])
        # 当split_2的列数大于1列时 直接添加
        if split_2.shape[1] > 1:
            results = results_temp.join(split_2[['作者机构' + str(index) + '_' + str(1), '作者机构' + str(index) + '_'  + str(2)]])
            df = results
        else:
            # 当split_2的列数只有1列时，在末尾添加空列
            results = results_temp.join(split_2['作者机构' + str(index) + '_' + str(1)])
            results['作者机构' + str(index) + '_' + str(2)] = None
            df = results
# 删除先前的作者-机构所有列，范围:y-x
df = df.drop(df.columns[:32], axis=1)
data = data.drop(data.columns[12:44], axis=1)
data_final = data.join(df)
# datas = data.join(df)

