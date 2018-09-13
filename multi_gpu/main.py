# coding:utf-8

import os
import json
import numpy as np
import tensorflow as tf
from time import time
from keras.layers import Input
from keras.utils import to_categorical
from models.wide_resnet import TestModelBuild
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, Callback
from opt.cycle_lr import SGDRScheduler


class TimeLog(Callback):
    def on_batch_begin(self, batch, logs={}):
        self.start = time()

    def on_batch_end(self, batch, logs={}):
        self.end = time()
        diff = self.end - self.start
        self.timelog.append(diff)
        self.start

    def on_epoch_begin(self, epoch, logs={}):
        self.timelog = []

    def on_epoch_end(self, epoch, logs={}):
        print(np.mean(self.timelog))


def schedule(epoch, lr):
    d = dict({60: 2e-2,
              120: 4e-3,
              160: 8e-4})
    if epoch in d:
        return d[epoch]
    return lr


if __name__ == "__main__":

    gpu_count = 4

    builder = TestModelBuild(10, 28, 10)
    inputs = Input(shape=(32, 32, 3))

    with tf.device("/cpu:0"):
        base_model = builder.build(inputs)

    model = multi_gpu_model(base_model, gpus=gpu_count)

    BASE_DIR = "../data/"

    X_train = np.load(os.path.join(BASE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))

    mean = X_train.mean(axis=(0, 1, 2))
    std = X_train.std(axis=(0, 1, 2))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_class = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_class)
    y_test = to_categorical(y_test, num_classes=num_class)

    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(sgd, loss="categorical_crossentropy", metrics=["acc"])

    data_size = X_train.shape[0]
    epochs = 200
    batch_size = 128 * gpu_count
    cycle = SGDRScheduler(min_lr=1e-4,
                          max_lr=2e-2,
                          steps_per_epoch=np.ceil(data_size/batch_size),
                          lr_decay=0.9,
                          cycle_length=1170,
                          mult_factor=1.5)
    callbacks = [TimeLog()]
    callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))
    gen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125)

    gen.fit(X_train)
    history = model.fit_generator(gen.flow(X_train,
                                  y_train,
                                  batch_size=batch_size),
                                  steps_per_epoch=(X_train.shape[0] // batch_size),
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=(X_test, y_test))

    base_model.save("test-model.hdf5")
    json.dump(history.history, open("history.json", "w"))
