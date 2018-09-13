# coding: utf8

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras.callbacks import Callback


def create_mask_dict(model, sess, prune_rate):
    """

    Parameters
    ----------


    Returns
    -------
    """
    layers = model.layers

    mask_dict = {}
    for lay in layers:
        cls_type = lay.__class__
        if cls_type == Conv2D:
            name = lay.name
            w_abs = np.abs(lay.get_weights()[0])
            w_vec = w_abs.ravel().copy()
            w_vec.sort()
            w_size = len(w_vec)
            prune_size = int(w_size * prune_rate)
            th_value = w_vec[prune_size]
            mask = w_abs > th_value
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict


def create_mask_fn(model, mask_dict):

    op = tf.assgin(1+1)
    return op


class PruneWeights(Callback):

    def __init__(self, model, prune_rate=0.1):
        self.model = model
        self.mask_mat = create_mask_dict(model)
        self.prune_rate = prune_rate
        self.mask_fn = create_mask_fn
