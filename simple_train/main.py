# coding:utf-8

import os
import imp
import json
import shutil
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from time import time
from keras.layers import Input
from keras.utils import to_categorical
from models.wide_resnet import ModelBuild
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint


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


class CustomLRSchedule(object):

    def __init__(self, d):
        self.d = d

    def schedule(self, epoch, lr):
        if epoch in self.d:
            return self.d[epoch]
        return lr


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--configpath')
    parser.add_argument('-o', '--outputdir')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # コマンドライン引数を取得
    args = parse()

    config_path = args.configpath
    output_path = args.outputdir

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config = imp.load_source("config", config_path)
    d = config.config_dict

    lr = d["lr"]
    max_epoch = d["max_epoch"]
    batch_size = d["batch_size"]

    if d["lr_ctl"]["type"] == "sch":
        schedule_dict = d["lr_ctl"]["sch"]
        lr_builder = CustomLRSchedule(schedule_dict)
        schedule = lr_builder.schedule
    elif d["lr_ctl"]["type"] == "XXXX":
        # TODO cyclical learning rateを実装する予定．
        pass

    data_dir = d["data_dir"]

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    mean = X_train.mean(axis=(0, 1, 2))
    std = X_train.std(axis=(0, 1, 2))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_class = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_class)
    y_test = to_categorical(y_test, num_classes=num_class)

    builder = ModelBuild(num_class, 28, 10)
    inputs = Input(shape=(32, 32, 3))
    model = builder.build(inputs)

    if d["optimizer"] == "sgd":
        opt = SGD(lr=lr, momentum=0.9, nesterov=True)
    elif d["optimizer"] == "rmsprop":
        pass
    elif d["optimizer"] == "adam":
        pass

    model.compile(opt, loss="categorical_crossentropy", metrics=["acc"])

    conf_name = config_path.split("/")[-1]
    shutil.copy(config_path, os.path.join(output_path, conf_name))

    data_size = X_train.shape[0]
    batch_size = batch_size
    callbacks = []
    callbacks.append(LearningRateScheduler(schedule=schedule, verbose=1))
    callbacks.append(ModelCheckpoint(os.path.join(output_path, "best_acc_model.hdf5"),
                                     monitor="val_acc",
                                     verbose=1,
                                     save_best_only=True))

    gen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125)

    gen.fit(X_train)
    steps_per_epoch = X_train.shape[0] // batch_size
    model.save(os.path.join(os.path.join(output_path, "init-model.hdf5")))
    history = model.fit_generator(gen.flow(X_train,
                                  y_train,
                                  batch_size=batch_size),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=max_epoch,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=(X_test, y_test))

    model.save(os.path.join(output_path, "last-model.hdf5"))
    json.dump(history.history, open(os.path.join(output_path, "history.json"), "w"))
