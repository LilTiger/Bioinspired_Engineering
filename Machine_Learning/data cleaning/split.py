# 此操作在运行screen comma之后 用于将作者/机构/国家分列
# 切片范围以x,y的形式给出 请仔细观察切片代码上方对应注释内容
# 除此之外还包括删除行/列的方法 注意学习
import pandas as pd

# 读取数据
data = pd.read_csv('scopus - heart.csv')
# 删除首行多余的列【原因是直接从scopus-temp中再次读出，会默认再加入一列】
data.drop(data.columns[0], axis=1, inplace=True)
# 取作者-机构所有列（形式为data.iloc[:, x:y])
df = data.iloc[:, 11:43].copy()
df.drop([0], inplace=True)  # 去掉首行

# range的值即为df的所有列范围（索引从0开始）【range的值为0, y-x】
for index in range(0, 32):

        split_1 = df.iloc[:, index].astype('str').str.split(', ', 1, expand=True)
        # 一定注意代码的规范 多输出一个[]就会使columns的格式从index变为multiindex 进而导致rename失败
        split_1.columns = ['作者机构' + str(index) + '_' + str(i) for i in split_1.columns]
        split_2 = split_1['作者机构' + str(index) + '_' + str(1)].str.rsplit(', ', 1, expand=True)
        split_2.columns = ['作者机构' + str(index) + '_' + str(i+1) for i in split_2.columns]

        # 加入split_1中分割的作者和split_2中分割的机构和国家
        results_temp = df.join(split_1['作者机构' + str(index) + '_' + str(0)])
        results_temp.rename(columns={'作者机构' + str(index) + '_' + str(0): '作者' + str(index)}, inplace=True)
        # 当split_2的列数大于1列时 直接添加
        if split_2.shape[1] > 1:
            results = results_temp.join(split_2[['作者机构' + str(index) + '_' + str(1), '作者机构' + str(index) + '_' + str(2)]])
            results.rename(columns={'作者机构' + str(index) + '_' + str(1): '机构' + str(index)}, inplace=True)
            results.rename(columns={'作者机构' + str(index) + '_' + str(2): '国家' + str(index)}, inplace=True)
            df = results
        else:
            # 当split_2的列数只有1列时，在末尾添加空列
            results = results_temp.join(split_2['作者机构' + str(index) + '_' + str(1)])
            results.rename(columns={'作者机构' + str(index) + '_' + str(1): '机构' + str(index)}, inplace=True)
            results['国家' + str(index)] = None
            df = results
# 删除df中先前的作者-机构所有列，范围:y-x
df = df.drop(df.columns[:32], axis=1)
# 删除data中先前的作者-机构所有列，范围x:y
data = data.drop(data.columns[11:43], axis=1)
data_final = data.join(df)

# 当直接读取CSV文件没有列名 后续想为列名赋值时 可使用下列方法 【注意columns的值需严格按照scopus.csv中格式或顺序修改】
# 按照原始格式赋予列名，范围：[x:]
data_final.columns = ['副主题', '作者', '作者ID', '标题', '年份', '来源出版物名称', '引用量',
                      'DOI', '关键词', '摘要', '归属机构'] + data_final.columns[11:].tolist()
data_final.drop([0], inplace=True)  # 去掉首行

data_final.to_csv('scopus - result.csv', encoding='utf-8-sig')


