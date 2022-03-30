# 卷积计算公式:
# N = （W-K+2P）/ S +1
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# 注意 使用cuda时 需要将 模型 和 图像和标签 都转到cuda中运行 可用.to(device)
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
# torch.cuda.empty_cache()
root = os.getcwd() + './data_pre/'


class CustomDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        super(CustomDataset, self).__init__()
        file = open(txt_path, 'r')
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
        img = Image.open(path).convert('RGB')
        # 如果传递了transform的参数 则按照传递的参数执行
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
train_data = CustomDataset(txt_path=root+'train.txt', transform=transform_method)
test_data = CustomDataset(txt_path=root+'text.txt', transform=transform_method)
# 使用DataLoader封装数据集 可以mini_batch形式输入 可shuffle 可使用多线程加速
# 目的是与 深度学习 方法进行关联
# 若想使用多线程加速 即(num_workers不为0) 记得在edit configuration中 取消勾选 Run with Python Console
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=8)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.fc1 = nn.Linear(53*53*64, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 53*53*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(dataloader, model, loss_function, optimizer):
    # 过程简单 大体流程是:预测数据 计算损失 梯度清零 反向传递 梯度更新 打印参数 保存模型
    # 封装好的train_loader和train_data并无二异 可直接作为数据集使用
    # 此处的 x 和 y 分别代表 train_loader中提取出的 图片路径 和 标签
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = loss_function(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            # 注意format的用法
            print("loss : {},  [{}/{}]".format(loss.item(), batch * len(x), size))

    # torch.save(self.model.state_dict(), 'linear.pth')


def tes_loop(dataloader, model, loss_function):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 测试时无需计算梯度 提高计算效率
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            # .item()可以获取元素 比如一维张量使用item()转换为float数值
            test_loss += loss_function(prediction, y).item()
            # argmax(1)在 二维数组 中的作用是 返回 每一行 最大元素的 索引
            # 此model中 prediction为 类别数*batch_size 的Tensor 每一行代表每一类的概率 argmax(1)会输出概率最大的类别 对应的索引
            # 因此 类文件夹务必从0开始编号
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    print(f"Accuracy : {100*(correct/size):>0.1f}%, Average loss : {(test_loss/num_batches):>8f}")


model = NeuralNetwork().to(device)
# 超参数经常定义在全局
learning_rate = 0.0001
epochs = 1000
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        tes_loop(test_loader, model, loss_fn)
    print("Done!")
