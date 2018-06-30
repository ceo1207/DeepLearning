# logistic regression examples for keras using mnist dataset
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.datasets import mnist
# load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()
# (60000, 28, 28) (60000,)

# preprocess data
from keras.utils import to_categorical
train_x = train_x.reshape(train_x.shape[0], 28*28)
train_x = train_x.astype('float32')
train_x = train_x/255   # normalization
train_y_one_hot = to_categorical(train_y)

test_x = test_x.reshape(test_x.shape[0], 28*28)
test_x = test_x.astype('float32')
test_x = test_x/255   # normalization
test_y_one_hot = to_categorical(test_y)

# create the network
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Dropout
# do not need consider the dim for samples
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y_one_hot, epochs=5, batch_size=128, validation_data=(test_x, test_y_one_hot))
result = model.evaluate(test_x, test_y_one_hot)
print result

# to see the training performance
# keys ['acc', 'loss', 'val_acc', 'val_loss']
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_loss'], 'r')

plt.figure()
plt.plot(history.history['acc'], 'b')
plt.plot(history.history['val_acc'], 'r')

# visually test
test_data = test_x[[0], :]
labelPre = model.predict(test_data.reshape(1, 784))
import numpy as np
print np.argmax(labelPre)
plt.imshow(test_data.reshape(28, 28))

model.save('mnist0630.h5')
