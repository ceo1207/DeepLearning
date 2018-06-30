# use a pre-trained model file to predict
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import load_model
predict_model = load_model('mnist0630.h5')
from keras.datasets import mnist
(_, _), (test_x, test_y) = mnist.load_data()
label = predict_model.predict(test_x[[0], :].reshape(1, 784))
import matplotlib.pyplot as plt
plt.imshow(test_x[0])
import numpy as np
print np.argmax(label)
