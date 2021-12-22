# coding=utf-8
# author: kayi
# logistic regression examples for keras using mnist dataset
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
# 需要使用这个后端，才能正常显示
matplotlib.use('TkAgg')
print(matplotlib.get_backend())
# tensorflow会占满所有显卡的显存，在跑代码之前，需要设置可见的GPU，以使用多块GPU中的某一块。
# 多个GPU之间可能不能并发，设置一块使用即可。不然会占用掉所有GPU的显存，影响别的训练进程。
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess_data(x, y):
    # 将训练数据x normalization
    x = x.reshape(x.shape[0], 28*28) # 784
    x = x.astype('float32')
    x = x/255.0
    x = x/255.0
    # 将数字的训练数据y 转化为one hot编码
    train_y_one_hot = to_categorical(y)
    return x, train_y_one_hot

def train_model():
    # create the network
    from keras.models import Sequential
    model = Sequential()

    from keras.layers import Dense, Dropout
    # do not need consider the dim for samples
    model.add(Dense(512, input_shape=(784,), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25)) # 设置丢弃，防止过拟合
    model.add(Dense(10, activation='softmax')) # softmax的激活函数 用于多分类问题的输出处理

    # 损失函数采用交叉熵 优化方法采用rmsprop，梯度下降的一些变种
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=5, batch_size=128, validation_data=(test_x, test_y))
    result = model.evaluate(test_x, test_y)
    print(result)

    # to see the training performance
    plt.figure()
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_loss'], 'r')

    plt.figure()
    plt.plot(history.history['accuracy'], 'b')
    plt.plot(history.history['val_accuracy'], 'r')

    # 训练之后的模型参数保存和加载
    model.save('mnist0630.h5')

def test_model():
    from keras import models
    model = models.load_model('mnist0630.h5')
    # visually test
    test_data = test_x[[0], :]
    labelPre = model.predict(test_data.reshape(1, 784))
    import numpy as np
    print(np.argmax(labelPre))  # argmax 求最大数的索引。很有用的函数，相当于是大规模的min max函数了
    plt.figure()
    plt.imshow(test_data.reshape(28, 28))
    # plt.savefig('myfig')
    # 需要添加如下，才能正常显示图片
    import pylab
    pylab.show()

if __name__ == '__main__':
    # load data
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # train (60000, 28, 28)   [0,255]     result (60000,)  [0,9]
    # print(train_x.shape[0])
    # 数据预处理
    train_x, train_y = preprocess_data(train_x, train_y)
    test_x, test_y = preprocess_data(test_x, test_y)
    # 训练模型
    train_model()
    # 可视化运行模型
    test_model()
