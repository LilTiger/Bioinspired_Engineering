# 反应扩散系统的动态绘制
# 包含生成gif动图的方法
"""
File Name:    Gray-Scott.py
Author:       Wang_Huiyu
Email:        huiyuwang001@163.com
Time:         2021/3/12 10:52
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import moviepy.editor as mp
import glob
import os
from PIL import Image

def my_laplacian(In):
    Out = -1.0 * In + 0.20 * (
                np.roll(In, 1, axis=1) + np.roll(In, -1, axis=1) + np.roll(In, 1, axis=0) + np.roll(In, -1, axis=0)) + \
          0.05 * (np.roll(np.roll(In, 1, axis=0), 1, axis=1) + np.roll(np.roll(In, -1, axis=0), 1, axis=1) + np.roll(
        np.roll(In, 1, axis=0), -1, axis=1) + np.roll(np.roll(In, -1, axis=0), -1, axis=1))
    return Out


def initial_conditions(n):
    A = np.ones((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    t = 0

    for i in range(50, 60):
        for j in range(50, 70):
            B[i, j] = 1

    for i in range(60, 80):
        for j in range(70, 80):
            B[i, j] = 1

    return t, A, B


workdir = str(os.getcwd() + '/')  # 为所有png和视频设置一个工作目录

if __name__ == "__main__":
    f = 0.054  # 进料率
    k = 0.064  # 去除率
    da = 1.0  # U的扩散率
    db = 0.5  # V的扩散率
    width = 128  # 网格大小
    dt = 0.25  # 每进行一需要0.25秒
    stoptime = 15000.0  # 一共有4000秒模拟时间




    t, A, B = initial_conditions(width)

    nframes = 1

    while t < stoptime:
        anew = A + (da * my_laplacian(A) - A * (B * B) + f * (1 - A)) * dt
        bnew = B + (db * my_laplacian(B) + A * (B * B) - (k + f) * B) * dt

        A = anew
        B = bnew
        t = t + dt
        # 每二十帧保存一次状态截图 作为拼接成动图的素材
        if nframes % 20 == 0:
            fig, ax0 = plt.subplots()
            fs = ax0.pcolor(B)
            fig.colorbar(fs, ax=ax0)
            TileName = (workdir + 'coo/' + str(nframes) + '.png')  # 传入的n表示n步，这里用来给生成的图像命名
            fig.savefig(TileName, dpi=100)
        nframes = nframes + 1

    frames = []
    imgs = sorted(glob.glob('coo/*.png'), key=os.path.getmtime)
    for i in imgs:
        temp = Image.open(i)
        keep = temp.copy()
        frames.append(keep)
        temp.close()

    # 删除全部的.png图
    for i in imgs:
        os.remove(i)

        # 片段合并保存为gif文件，永远循环播放
    frames[0].save('coo/coordinates.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=30, loop=0)

    # 转化为gif动图
    clip = mp.VideoFileClip("coo/coordinates.gif")


