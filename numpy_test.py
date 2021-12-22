# coding=utf-8
# author: kayi
# date: 2021/12/22
'''
numpy 练习
'''
import random
import numpy as np

# 创建矩阵
tmp = np.eye(4)
print(tmp, tmp.shape)
tmp = np.random.random([3, 4])
print(tmp)

# 矩阵的一些连接、复制操作
print(np.repeat(np.array([1, 2]), 3))
print(np.concatenate([np.array([1, 2]), np.array([3, 4])]))

tmp = np.arange(12)
# reshape时，从原n维数组最底层的维度遍历，由右到左依次构成新数组的维度，如(3，4)，先构建长度4的维度，而后使用3个数组构建第一维度
tmp = tmp.reshape(3, 4)

# 矩阵运算
# 转置运算
print(tmp.T)
print(tmp)
