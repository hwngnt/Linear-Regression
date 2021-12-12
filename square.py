# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:10:04 2021

@author: Hung Thanh Nguyen
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#numOfPoint = 30
#noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
#x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
#N = x.shape[0]
#y = 15*x + 8 + 20*noise
#plt.scatter(x, y)

data = pd.read_csv('data_square.csv').values
N = data.shape[0]
x2 = data[:, 0].reshape(-1, 1)
x = x2
y = data[:, 1].reshape(-1, 1)
plt.scatter(x2, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')
x2 = (x2 - x2.mean())/x2.std()
# y = y - y.mean()/y.std()
# x2 = np.hstack((np.ones((N, 1)), x2))
x2 = np.concatenate((np.ones((N, 1)), x2), axis=1)
x2_squared = np.square(x2[:,1:])
x2 = np.concatenate((x2, x2_squared), axis=1)
# w = np.array([-1.,0.,1.]).reshape(-1,1)
w = np.random.rand(3,1)
numOfIteration = 500
cost = np.zeros((numOfIteration,1))
learning_rate = 0.01
for i in range(1, numOfIteration):
    r = np.dot(x2, w) - y
    cost[i] = 0.5*np.sum(r*r)/N
    print(cost[i])
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, x2[:,1].reshape(-1,1)))
    # correct the shape dimension
    w[2] -= learning_rate*np.sum(np.multiply(r, np.square(x2[:,2].reshape(-1,1))))
    # print(cost[i])
# x0 = np.linspace(30,100,10000)
y0 = np.dot(x2, w)
plt.plot(x, y0, 'r')
plt.show()
# print(np.dot(x2, w))
# x1 = 50
# y1 = w[0] + w[1] * 50**2
# print('Giá nhà cho 50m^2 là : ', y1)

