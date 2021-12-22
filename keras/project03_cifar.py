# coding=utf-8
# author: kayi
# use cnn to classify cifar-10

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
import numpy as np
from keras.datasets import cifar10
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def preprocess_data(x, y):
    from keras.utils.np_utils import to_categorical
    x = x.astype('float32')
    x = x / 255  # normalization
    y_one_hot = to_categorical(y)
    return x, y_one_hot



def create_model():

    model = Sequential()
    # 32 filters 得到32维的卷积结果    the size of each filter 3*3*input_channel
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 卷积层与全连接层之间的衔接
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))   # 还是一样，多分类的问题，使用softmax

    # print information of the model
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, data):
    from keras.callbacks import TensorBoard
    # use tensorboard
    tensorboard = TensorBoard(log_dir='log')
    callback_lists = [tensorboard]

    train_x, train_y_one_hot, test_x, test_y_one_hot = data
    history1 = model.fit(train_x, train_y_one_hot, epochs=5, batch_size=128, validation_data=(test_x, test_y_one_hot),
                         callbacks=callback_lists)
    result = model.evaluate(test_x, test_y_one_hot)
    print(result)

    plt.figure()
    plt.plot(history1.history['loss'], 'b')
    plt.plot(history1.history['val_loss'], 'r')

    plt.figure()
    plt.plot(history1.history['accuracy'], 'b')
    plt.plot(history1.history['val_accuracy'], 'r')

    # in the training progress, you can change the learning rate, batch size to get better performance
    # history2 = model.fit(train_x, train_y_one_hot, epochs=5, batch_size=256, validation_data=(test_x, test_y_one_hot))
    # result = model.evaluate(test_x, test_y_one_hot)
    #
    # plt.figure()
    # plt.plot(history2.history['loss'], 'b')
    # plt.plot(history2.history['val_loss'], 'r')
    # plt.title("loss")
    # plt.figure()
    # plt.plot(history2.history['acc'], 'b')
    # plt.plot(history2.history['val_acc'], 'r')
    # plt.title("accuracy")

    import pylab
    pylab.show()
    model.save('my_cifar_10_model.h5')


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    # (50000, 32, 32, 3) (50000, 1)   (10000, 32, 32, 3)
    # 不同于mnist， 结果虽然只有一个数，但还是构成了一维
    print(train_x.shape)

    # Find the unique numbers from the train labels
    classes = np.unique(train_y) # [0 1 2 3 4 5 6 7 8 9]
    nClasses = len(classes)

    train_x, train_y = preprocess_data(train_x, train_y)
    test_x, test_y_one_hot = preprocess_data(test_x, test_y)

    my_model = create_model()
    data = [train_x, train_y, test_x, test_y_one_hot]
    train_model(my_model, data)
