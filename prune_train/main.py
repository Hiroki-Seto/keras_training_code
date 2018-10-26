# coding: utf8

import os
import imp
import json
import shutil
import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from models.wide_resnet import ModelBuild
from opt.prune_mask import PruneWeights
from opt.freeze_layer import freeze_layer


def CustopmLRSchedule(object):

    def __init__(self, d):
        self.d = d

    def chedule(self, epoch, lr):
        if epoch in self.d:
            return self.d[epoch]
        return lr


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse()

    config_path = args.config_path
    output_path = args.output_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config = imp.load_source("config", config_path)
    d = config.config_dict

    lr = d["lr"]
    max_epoch = d["max_epoch"]
    batch_size = d["batch_size"]

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

    if d["optimizer"] == "sgd":
        opt = SGD(lr=lr, momentum=0.9, nesterov=True)
    else:
        pass

    init_modelpath = d["init_modelpath"]
    if init_modelpath == "":
        mbuilder = ModelBuild(num_class, 28, 10)
        inputs = Input((32, 32, 3))
        model = mbuilder(inputs)
    else:
        model = load_model(init_modelpath)
    # change the trainable flag
    freeze_cls_type = d["freeze_cls_type"]
    freeze_layer(model, freeze_cls_type)
    model.compile(opt, loss="categorical_crossentropy", metrics=["acc"])

    conf_name = config_path.split("/")[-1]
    shutil.copy(config_path, os.path.join(output_path, conf_name))

    data_size = len(X_train)
    sess = K.get_session()
    timing = d["timing"]
    prune_rate = d["prune_rate"]
    prune_type = d["prune_type"]
    callbacks = []
    callbacks.append(PruneWeights(model,
                                  sess,
                                  prune_type=prune_type,
                                  timing=timing,
                                  prune_rate=prune_rate))
    gen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125)

    gen.fit(X_train)
    steps_per_epoch = data_size // batch_size
    history = model.fit_generator(gen.flow(X_train, y_train, batch_size),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=max_epoch,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=(X_test, y_test))
    callbacks[0].call()
    model.save(os.path.join(output_path, "last_model.hdf5"))
    with open(os.path.join(output_path, "history.json"), "w") as f:
        json.dump(history.history, f)

    # check accuracy with after prune.
    preds_prob = model.predict(X_test)
    preds = np.argmax(preds_prob, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    match = (y_true == preds)
    acc = np.sum(match) / len(y_true)
    print("Pruning Accuracy: {0}".format(acc))
