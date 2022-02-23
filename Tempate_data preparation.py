import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 有两种方式使用自己的数据集
# 按照传统计算机的思维 请务必保证类文件夹从0开始编号

# 此种通过重构DataSet函数的好处是 训练、验证、测试集的 每一类 的数据 都可放于同一文件夹
# 由train.txt val.txt test.txt指定读取即可
# 缺点是 需要首先构建图片路径和标签的text文件 需要重写DataSet类中的三个函数
##############################
root = os.getcwd() + './data_pre/'


class CustomDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        super(CustomDataset, self).__init__()
        file = open(txt, 'r')
        img_list = []
        for line in file:
            # line.strip('\n')
            line.rstrip('\n')
            lines = line.split()
            img_list.append((lines[0], int(lines[1])))

        self.img_list = img_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, label = self.img_list[index]
        img = Image.open(path).convert('rgb')
        # 如果传递了transform的参数 则按照传递的参数执行 如后文的ToTensor()方法
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)


transform_method = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 实例化训练集/测试集对象 此为真正制造train/test数据集的操作
train_data = CustomDataset(txt=root+'train.txt', transform=transform_method)
test_data = CustomDataset(txt=root+'text.txt', transform=transform_method)
# 使用DataLoader封装数据集 可以mini_batch形式输入 可shuffle 可使用多线程加速
# 目的是与 深度学习 方法进行关联
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
text_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0)

print(len(train_data))
print(len(test_data))
######################################


# 使用ImageFolder可直接获取文件夹中的数据
# 需要将数据分别存放在训练/验证/测试集对应的类文件夹内
# 由构造数据的class_to_idx知 按照类文件夹名称 ImageFolder函数 顺序 赋予类别
# 因此若将类文件夹 从0开始命名 如此便可与ImageFolder方法中的类别标签一致
######################################
from torchvision import datasets
transform_method = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 实例化训练集/测试集对象 此为真正制造train/test数据集的操作
train_data = datasets.ImageFolder(root+'/train/', transform=transform_method)
test_data = datasets.ImageFolder(root+'/test/', transform=transform_method)
# 使用DataLoader封装数据集 可以mini_batch形式输入 可shuffle 可使用多线程加速
# 目的是与 深度学习 方法进行关联
trainloader = DataLoader(dataset=train_data, batch_size=6, shuffle=True, num_workers=4)
textloader = DataLoader(dataset=test_data, batch_size=6, shuffle=True, num_workers=4)

print(len(train_data))
print(len(test_data))
