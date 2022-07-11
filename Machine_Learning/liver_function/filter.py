# 此filter实现对scopus下载到的原始csv文件的作者-机构列进行数据格式化处理
# 具体操作为：去掉该列中 作者姓名中间多余的逗号；将作者、机构和国家之间以*号来分隔
# 此模板亦可泛化到类似操作 但切记 【替换后的符号】不可在原始数据 列 中检索到 否则会造成不准确
import csv
import pandas as pd

filename = 'scopus - pre.csv'
data = []
with open(filename, encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:

        temp = row[12].split(';')
        for index in range(temp.__len__()):
            # 去掉在字符串中第一个出现的 作者名之间的逗号
            temp[index] = temp[index].strip().replace(',', '', 1)
            # 更改作者和机构之间的标识符
            temp[index] = temp[index].replace(',', '★', 1)
            result = temp[index].rsplit(', ', 1)
            # 更改机构和国家之间的标识符
            temp[index] = '★ '.join(result)
        ssr = ';'.join(temp)
        row[12] = ssr
        print(ssr)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv('scopus - filter.csv', encoding='utf-8-sig', header=None, index=None)  # 保存到csv文件时不保存行索引和列索引

