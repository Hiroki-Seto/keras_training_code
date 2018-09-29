# coding: utf8

from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Dense, AveragePooling2D, MaxPool2D, Flatten
from keras.models import Model
from keras import regularizers

decay_val = 5e-4


class ModelBuild():

    def __init__(self, num_class):
        self.num_class = num_class

    def ConvBN(self, x, filters, kearnel_size, name):
        x = Conv2D(filters=filters,
                   kernel_size=kearnel_size,
                   strides=(1, 1),
                   kernel_regularizer=regularizers.l2(decay_val),
                   padding="same",
                   name=name)(x)
        x = BatchNormalization(name="{0}_bn".format(name))(x)
        x = Activation("relu", name="{0}_act".format(name))(x)
        return x

    def build(self, inputs):

        x = self.ConvBN(inputs, 64, 3, "conv1_1")
        x = self.ConvBN(x, 64, 3, "conv1_2")
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self.ConvBN(x, 128, 3, "conv2_1")
        x = self.ConvBN(x, 128, 3, "conv2_2")
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self.ConvBN(x, 256, 3, "conv3_1")
        x = self.ConvBN(x, 256, 3, "conv3_2")
        x = self.ConvBN(x, 256, 3, "conv3_3")
        x = self.ConvBN(x, 256, 3, "conv3_4")
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self.ConvBN(x, 256, 1, "conv4_1")
        x = self.ConvBN(x, 256, 1, "conv4_2")
        x = self.ConvBN(x, 256, 1, "conv4_3")
        x = self.ConvBN(x, 256, 1, "conv4_4")

        x = AveragePooling2D(pool_size=(4, 4))(x)
        x = Flatten()(x)
        x = Dense(self.num_class,
                  activation="softmax",
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(decay_val))(x)
        model = Model(inputs, x)
        return model
