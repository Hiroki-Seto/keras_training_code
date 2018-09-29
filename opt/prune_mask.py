# coding: utf8

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense
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
        if (cls_type == Conv2D) or (cls_type == Dense):
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
    """

    Parameters
    ----------

    Returns
    -------

    """

    layers = model.layers

    ops = []
    for lay in layers:
        name = lay.name
        if name in mask_dict:
            mask = mask_dict[name]
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            w = lay.weights[0]
            fn = w * mask
            assign = tf.assign(w, fn)
            ops.append(assign)
    return ops


class PruneWeights(Callback):
    """
    """

    def __init__(self, model, sess, timing=10, prune_rate=0.1):
        """
        Parameters
        ----------
        """
        self.model = model
        self.sess = sess
        self.timing = timing
        self.prune_rate = prune_rate
        self.mask_dict = create_mask_dict(model, sess, prune_rate)
        self.mask_fn = create_mask_fn(model, self.mask_dict)

    def call(self):
        """
        """
        self.sess.run(self.mask_fn)

    def on_epoch_begin(self, epoch, logs={}):
        """
        """
        if (epoch % self.timing == 0):
            print("Prune")
            self.sess.run(self.mask_fn)
