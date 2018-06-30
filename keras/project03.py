# use cnn to classify cifar-10
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from keras.datasets import cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
# (50000, 3, 32, 32) (50000, 1)   (10000, 3, 32, 32)
# Find the unique numbers from the train labels
classes = np.unique(train_y)
nClasses = len(classes)

from keras.utils import to_categorical
train_x = train_x.astype('float32')
train_x = train_x/255   # normalization
train_y_one_hot = to_categorical(train_y)

test_x = test_x.astype('float32')
test_x = test_x/255   # normalization
test_y_one_hot = to_categorical(test_y)

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

model = Sequential()
# 32 filters     the size of each filter 3*3*input_channel
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(train_x.shape[1:])))
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

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# print information of the model
model.summary()

from keras.callbacks import TensorBoard
# use tensorboard
tensorboard = TensorBoard(log_dir='log')
callback_lists = [tensorboard]


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_x, train_y_one_hot, epochs=5, batch_size=128, validation_data=(test_x, test_y_one_hot), callbacks=callback_lists)
result = model.evaluate(test_x, test_y_one_hot)
print result

import matplotlib.pyplot as plt
plt.figure()
plt.plot(history1.history['loss'], 'b')
plt.plot(history1.history['val_loss'], 'r')

plt.figure()
plt.plot(history1.history['acc'], 'b')
plt.plot(history1.history['val_acc'], 'r')

history2 = model.fit(train_x, train_y_one_hot, epochs=5, batch_size=256, validation_data=(test_x, test_y_one_hot))
result = model.evaluate(test_x, test_y_one_hot)
print result

plt.figure()
plt.plot(history2.history['loss'], 'b')
plt.plot(history2.history['val_loss'], 'r')
plt.title("loss")
plt.figure()
plt.plot(history2.history['acc'], 'b')
plt.plot(history2.history['val_acc'], 'r')
plt.title("accuracy")

# in the training progress, you can change the learning rate, batch size to get better performance
