# coding=utf-8
# author: kayi
# date: 2021/12/22
'''
1. 将人脸识别视为一个多分类问题
https://blog.csdn.net/hahajinbu/article/details/72877998
2. 大数据集下，将人脸量化为坐标，比较最近的点
关键问题是，这样的抽象，如何将距离这个作为loss，梯度又如何传递
'''
import scipy.io as sio
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import seaborn
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
import keras.optimizers as optimizers
from keras.utils import np_utils

# 实际训练数据标签是1-68，,68个类
nb_classes = 69  # 添加一个虚类0，这样不用labels可以直接用，不用每个类编号-1。
nb_epoch = 80  # 迭代周期，这是取的最大的
batch_size = 40  # 每批次样本数  就是输入网络第一维的数字

lr = 0.002  # 学习率 步长
decay = 1e-6  # 学习率衰减
momentum = 0.9  # 冲量

# input image dimensions
img_rows, img_cols = 64, 64  # 图片的行列数
# number of convolutional filters to use
nb_filters1, nb_filters2 = 5, 10  # 卷积核数目
# size of pooling area for max pooling
nb_pool = 2  # 池化大小
# convolution kernel size
nb_conv = 3  # 卷积核大小

trainData = []
trainLabels = []

testData = []
testLabels = []


def assembleData(filename):  # 读取一次数据并组合
    data = sio.loadmat(filename)
    train = data['fea']
    labels = data['gnd']
    istest = np.where(data['isTest'] == 1)
    testIndex = list(istest[0])
    t_labels = []  # 本次抽取的测试集
    t_train = []  # 本次抽取的测试标签
    for i in testIndex:  # 抽取测试集和测试标签
        t_train.append(train[i, :])
        t_labels.append(labels[i][0])
    train = np.delete(train, testIndex, axis=0)  # 从样本中删除测试集
    labels = np.delete(labels, testIndex, axis=0)  # 从标签中删除测试标签
    trainData.extend(train)
    trainLabels.extend(labels[:, 0])
    testData.extend(t_train)
    testLabels.extend(t_labels)


def splitData():
    # 划分训练集与验证集 非常有用
    train_x, val_x, train_y, val_y = train_test_split(trainData, trainLabels, test_size=0.3,
                                                      random_state=1)  # 划分训练集与验证集
    rval = [(train_x, train_y), (val_x, val_y)]
    return rval


def Net_model(lr=lr, decay=decay, momentum=momentum):  # 建立并编译模型
    model = Sequential()  # 贯序模型
    model.add(Conv2D(nb_filters1, (nb_conv, nb_conv),
                     padding='valid',
                     kernel_initializer='glorot_normal',
                     input_shape=(1, img_rows, img_cols), name='Conv2D_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000))  # Full connection
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_val, Y_val):  # 训练模型，返回训练过程记录
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2,
                        validation_data=(X_val, Y_val))
    return history


def getBesthistoryacc():  # 读取最好的结果
    read_file = open('bestacc.pkl', 'rb')
    best = pickle.load(read_file)
    read_file.close()
    return best


def saveHistory(history):  # 没啥卵用，打印history包括的东西的
    # write_his = open('history', 'wb')
    # pickle.dump(history.history['acc'], write_his, -1)
    # pickle.dump(history.history['val_acc'], write_his, -1)
    # write_his.close()
    print(history.history.keys())


def displayHistory(history, mode=1):  # 显示训练过程的loss变化
    print(history.history.keys())
    write_his = open('history', 'wb')
    if mode > 0:
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.xlabel('epoch')
        plt.legend(['acc', 'loss'], loc='upper left', fontsize='x-large')
    # summarize history for acc
    if mode > 1:  # 有验证集
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['val_loss'])
        plt.legend(['acc', 'loss', 'val-acc', 'val-loss'], loc='upper left', fontsize='x-large')

    plt.show()
    write_his.close()


def loadModelAndPredict():  # 加载已训练的模型权重并进行预测
    model.load_weights('model2_weights.h5')
    classes = model.predict_classes(X_test, verbose=1)
    test_accuracy = np.mean(np.equal(testLabels, classes))
    print("accuracy:", test_accuracy)


if __name__ == "__main__":
    filelist = ["PIE dataset\Pose05_64x64.mat", "PIE dataset\Pose07_64x64.mat", "PIE dataset\Pose09_64x64.mat",
                "PIE dataset\Pose27_64x64.mat", "PIE dataset\Pose29_64x64.mat"]
    for file in filelist:
        assembleData(file)
    trainData = np.array(trainData)
    trainLabels = np.array(trainLabels)
    testData = np.array(testData)
    testLabels = np.array(testLabels)

    (X_train, Y_train), (X_val, y_val) = splitData()
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  # 输入是样本数、通道数、行列数
    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
    X_test = testData.reshape(testData.shape[0], 1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_val /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(Y_train,
                                      nb_classes)  # 贯序模型多分类keras要求格式为binary class matrices,转化一下（就是转化为分类矩阵，Kij=1表示i个样本分为第j类）
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(testLabels, nb_classes)

    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)

    model = Net_model()
    history = train_model(model, X_train, Y_train, X_val, Y_val)

    classes = model.predict_classes(X_test, verbose=1)  # 立即预测结果
    test_accuracy = np.mean(np.equal(testLabels, classes))
    print("accuracy:", test_accuracy)
    if os.path.exists('bestacc.pkl'):  # 保存预测结果
        read_file = open('bestacc.pkl', 'rb')
        bestacc = pickle.load(read_file)
        read_file.close()
        if (bestacc < test_accuracy):
            write_file = open('bestacc.pkl', 'wb')
            pickle.dump(test_accuracy, write_file, -1)
            model.save_weights('model2_weights.h5', overwrite=True)
            write_file.close()
    else:
        write_file = open('bestacc.pkl', 'wb')
        pickle.dump(test_accuracy, write_file, -1)
        model.save_weights('model2_weights.h5', overwrite=True)
        write_file.close()

    displayHistory(history, mode=2)
    best = getBesthistoryacc()
    print("best accuracy:", best)

