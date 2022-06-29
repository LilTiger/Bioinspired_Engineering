# 分类图像
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
X,y = make_blobs(centers=2, center_box=(1, 10))
# 数据集绘图
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Albumin secreted on day 1', 'Albumin not secreted on day 1'], loc=4)
plt.xlabel('Cell Type')
plt.ylabel('Scaffold Type')
plt.savefig('1.tif')
plt.show()


# # sns.regplot用来观察 两个一维数据 （可等价为一元线性回归）的关联性 提供拟合曲线
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# csv_file = open('parameters1.csv')
# data = pd.read_csv(csv_file)
#
# # 指明csv文件中标签和特征列名即可
# label_name = 'Day 7 Albumin'
# feature_name = ['Cell Type', 'Seeding Density']
# x_label = data[feature_name]
# y_label = data[label_name]
#
# sns.regplot(x='Cell Type',y='Day 7 Albumin',data=data)
# plt.show()


# # 二元线性拟合图像
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # # 创建函数，用于生成不同属于一个平面的100个离散点
# # def not_all_in_plane(a, b, c):
# #     x = np.random.uniform(-10, 10, size=100)
# #     y = np.random.uniform(-10, 10, size=100)
# #     z = (a * x + b * y + c) + np.random.normal(-1, 1, size=100)
# #     return x, y, z
# #
# # # 调用函数，生成离散点
# # x, y, z = not_all_in_plane(2, 5, 6)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # 什么范围的数据用什么标志来标识
# for m, zlow, zhigh in [('o', 0, 35), ('^', 30, 60)]:
#     x = np.random.uniform(0, 10, size=100)
#     y = np.random.uniform(0, 10, size=100)
#     z = (2 * x + 5 * y + 6) + np.random.normal(-1, 1, size=100)
#     ax.scatter(x, y, z, marker=m)
#
# a = 0
# A = np.ones((100, 3))
# for i in range(0, 100):
#     A[i, 0] = x[a]
#     A[i, 1] = y[a]
#     a = a + 1
# # print(A)
#
# # 创建矩阵b
# b = np.zeros((100, 1))
# a = 0
# for i in range(0, 100):
#     b[i, 0] = z[a]
#     a = a + 1
# # print(b)
#
# # 通过X=(AT*A)-1*AT*b直接求解
# A_T = A.T
# A1 = np.dot(A_T, A)
# A2 = np.linalg.inv(A1)
# A3 = np.dot(A2, A_T)
# X = np.dot(A3, b)
# print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))
#
#
# xs = np.linspace(0, 10, 100)
# ys = np.linspace(0, 10, 100)
# xs, ys = np.meshgrid(xs, ys)
# zs = X[0, 0] * xs + X[1, 0] * ys + X[2, 0]
# ax.plot_wireframe(xs, ys, zs, rstride=100, cstride=100)
# ax.set_xlabel('Cell Type')
# ax.set_ylabel('Scaffold Type')
# ax.set_zlabel('Day 7 Albumin')
# plt.savefig('2.tif')
# plt.show()
