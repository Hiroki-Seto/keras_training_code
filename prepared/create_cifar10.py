import os
import numpy as np
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
BASE_DIR = "../data/"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

np.save(os.path.join(BASE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(BASE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)
