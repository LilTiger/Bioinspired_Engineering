# This is the template script generating the '.txt' file for the data
# 此种利用pytorch自带dataloader方法的好处是 训练、验证、测试集的 每一类 的数据 都可放于同一文件夹
# 由train.txt val.txt test.txt指定读取即可
# 若要控制训练/验证/测试集的数量或比例 修改 len(files)*{percentage}
import os
import tqdm
idx = 1
category = 6  # category为数据类别总数
train = open('./data_pre/train.txt', 'w+')
text = open('./data_pre/text.txt', 'w+')
for idx in range(1, category+1):
    dir = './data_pre/' + str(idx) + '/'  # 定位到该 类别 图片文件的地址
    label = idx
    # os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
    files = os.listdir(dir)  # 列出该类别文件夹下的 所有图片
    files.sort()  # 排序

    for i, file in tqdm.tqdm(enumerate(files)):
        # 此处控制 拿出总数据的70%作为训练集 剩下的作为测试集
        percentage = int(len(files) * 0.7)
        if i < percentage:
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            train.write(name)
            i = i + 1

        else:
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            text.write(name)
            i = i + 1
text.close()
train.close()
print("generation completed.")
