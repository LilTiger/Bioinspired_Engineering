# 此代码用来去除作者-机构列中 每个作者名中间多出的逗号
# 学习从源文件修改特定列数据后再写入到源/新文件的方法
# 多看注释
import csv
import pandas as pd

filename = 'scopus.csv'
data = []
# encoding方式为utf-8-sig可以避免dataframe在数据开头加入的\ufeff标识
with open(filename, encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
        # csv文件按行读取，一行一个列表，如果要获取某些特定行内容，可像下面方法直接指定
        # if csv_reader.line_num == 1:  # 第一行标题不加入data
        #     continue
        for index in range(len(row)):
            if index < 11:  # 只操作作者-机构列
                continue
                # 若字符串为空 跳过
            else:
                if len(row[index]) != 0:
                    ins = row[index].strip().split(',', 1)
                    if len(ins) != 1:  # 防止分号分割出来的作者形式不标准，比如分割出来一个邮箱..
                        new = ins[0] + ins[1]
                        row[index] = new
                        print(row[index])
            # 将去掉逗号的加入新数据列
        data.append(row)  # 也可以指定某几列加入到data数组中，形如row[5:]

    # data包含清洗好的作者-机构列和源文件中保持不变的所有列，据此创建dataframe并写入文件
    df = pd.DataFrame(data)
    df.to_csv('scopus - heart.csv', encoding='utf-8-sig')
