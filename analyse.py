import os
import matplotlib.pyplot as plt
from pylab import *

def read_file():
    dir_name = './data/Franz Schubert'
    files = os.listdir(dir_name)
    paths = []
    for file in files:
        path = os.path.join('.', 'data', 'Franz Schubert', file)
        path = path.replace('\\', '/')
        paths.append(path)

    data_set1 = []
    for path in paths:
        with open(path, 'r') as f:
            text = f.read().strip().split(' ')
            for item in text:
                item = int(item)
                if item == 129 or item == 128:
                    continue
                data_set1.append(item)

    data_set2 = []
    with open('./test.txt') as f:
        text = f.read().strip().split(' ')
        for item in text:
            item = int(item)
            if item == 128 or item == 129:
                continue
            for i in range(len(paths)):
                data_set2.append(item)

    return data_set1, data_set2

def draw():
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    data_set1, data_set2 = read_file()
    plt.figure()  # 初始化一张图
    x = []
    x.append(data_set1)
    x.append(data_set2)
    plt.hist(x, bins=75, range=(30, 105), rwidth=0.6)  # 直方图关键操作
    plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
    plt.xlabel('notes')
    plt.ylabel('times')
    plt.title(r'频数分布直方图（蓝色：original 橙色：hidden）')
    plt.plot()
    plt.show()

draw()