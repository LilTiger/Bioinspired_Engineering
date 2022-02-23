# This is the template script generating the '.txt' file for the data
# 此种通过重构DataSet函数的好处是 训练、验证、测试集的 每一类 的数据 都可放于同一文件夹
# 由train.txt val.txt test.txt指定读取即可
# 若要控制训练/验证/测试集的数量或比例 修改 len(files)*{percentage}
import os
import tqdm
idx = 0
category = 6  # category为数据类别总数
# w+ 打开一个文件进行读写 并从头开始编辑
file_train = open('./data_pre/train.txt', 'w+')
file_text = open('./data_pre/text.txt', 'w+')
for idx in range(0, category):
    # 数据存放格式为 /data_pre/下存放类文件夹 每个类文件夹下存放所有图片
    dir = './data_pre/' + str(idx) + '/'  # 定位到该 类别 图片文件的地址
    label = idx
    files = os.listdir(dir)  # 列出该类别文件夹下的 所有图片
    files.sort()  # 排序

    for i, file in tqdm.tqdm(enumerate(files)):
        # 此处控制 拿出总数据的70%作为训练集 剩下的作为测试集
        num_train = int(len(files) * 0.7)
        if i < num_train:
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            file_train.write(name)
            i = i + 1

        else:
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            file_text.write(name)
            i = i + 1
file_text.close()
file_train.close()
print("generation completed.")
