# !/usr/bin/env python
# coding:utf-8
# Author: Huangyu
import numpy as np
import matplotlib.pyplot as plt

#数据准备
x_data = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
#ydata = b + w*xdata

#误差计算
x = np.arange(-200,-100,1)#b
y = np.arange(-5,5,0.1)#w
Z = np.zeros((len(x),len(y)))
#np.meshgrid函数接受两个一维数组，并产生两个二维矩阵（对应于两个数组中所有的(x,y)对）
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0   #Z[j][i]计算出每个w,b对应的误差
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w*x_data[n]) ** 2
        Z[j][i] = Z[j][i]*1.0/len(x_data)

#梯度下降法
#ydata = b + w*xdata
 #初始化
b = -120
w = -4
lr = 0.0000001 #学习率
iteration = 100000 #循环次数

#保存w,b历史值
b_history = [b]
w_history = [w]

for i in range(iteration):
    b_grid = 0.0
    w_grid = 0.0
    for i in range(len(x_data)):
        #对w,b求偏导的过程  此处一定要是-.
        #通过分析可以判断要是y_data[i]>b + w*x_data[i],w、b要相应的增大
        b_grid = b_grid - 2.0 * (y_data[i] - b - w*x_data[i])*1.0
        w_grid = w_grid - 2.0 * (y_data[i] - b - w*x_data[i])*x_data[i]
    b = b - lr * b_grid
    w = w - lr * w_grid
    #保存w,b历史值
    b_history.append(b)
    w_history.append(w)
# print(w_history)
# print(b_history)
#呈现图像
#呈现Z数组
plt.contourf(x,y,Z,50,alpha = 0.5,cmap = plt.get_cmap('jet'))
#把最佳w,b呈现出来
plt.plot([-188.4],[2.67],'x',ms = 12,markeredgewidth = 3,color = 'orange')
#把w,b历史值呈现出来
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='blue')
plt.xlim = (-200,-100)
plt.ylim = (-5,5)
plt.xlabel(r'$b$',fontsize = 16)
plt.ylabel(r'$b$',fontsize = 16)
plt.show()

