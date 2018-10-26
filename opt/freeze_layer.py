# coding: utf8

from keras.layers import Conv2D, Dense


def freeze_layer(model, cls_type="Conv2D"):

    if cls_type == "Conv2D":
        target_cls = Conv2D
    elif cls_type == "Dense":
        target_cls = Dense

    layers = model.layers

    for lay in layers:
        cls = lay.__class__

        # Dense層が複数含まれる場合に対応していない
        if cls == target_cls:
            lay.trainable = False
