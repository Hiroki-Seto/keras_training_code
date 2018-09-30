# coding: utf8

from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Dense, Add, AveragePooling2D, Flatten
from keras.models import Model
from keras import regularizers

# 下記の値は，cifar10で精度を出した設定．
# delta = 5e-4
# cifar100用に強めてみる．
delta = 5e-4 * 2


class ModelBuild():

    def __init__(self, num_class, depth, width):
        self.num_class = num_class
        self.repeat = (depth - 4) // 6
        self.nfilter = [a*width for a in [16, 32, 64]]

    def block(self, inputs, stage, nrepeat, strides, nf, diffch=False):
        x = BatchNormalization(name="{0}_{1}_bn0".format(stage,
                                                         nrepeat))(inputs)
        x = Activation("relu", name="{0}_{1}_act0".format(stage,
                                                          nrepeat))(x)
        x = Conv2D(nf, 3, strides=strides, padding="SAME",
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(delta),
                   name="{0}_{1}_conv0".format(stage, nrepeat))(x)
        x = BatchNormalization(name="{0}_{1}_bn1".format(stage,
                                                         nrepeat))(x)
        x = Activation("relu", name="{0}_{1}_act1".format(stage,
                                                          nrepeat))(x)
        x = Conv2D(nf, 3, strides=(1, 1), padding="SAME",
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(delta),
                   name="{0}_{1}_conv1".format(stage, nrepeat))(x)
        if diffch:
            _x = Conv2D(nf, 1, strides=strides, padding="SAME",
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(delta),
                        name="{0}_{1}_convdiff".format(stage,
                                                       nrepeat))(inputs)
            x = Add()([x, _x])
        else:
            x = Add()([x, inputs])
        return x

    def repeat_block(self, inputs, stage, repeat, strides, nf):
        x = inputs
        for i in range(repeat):
            if i != 0:
                x = self.block(x, stage, i, (1, 1), nf, diffch=False)
            else:
                x = self.block(x, stage, i, strides, nf, diffch=True)
        return x

    def build(self, inputs):

        x = Conv2D(16, 3, strides=(1, 1), padding="SAME",
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(delta),
                   name="conv_first")(inputs)
        x = self.repeat_block(inputs=x,
                              stage="stage1",
                              repeat=self.repeat,
                              strides=(1, 1),
                              nf=self.nfilter[0])
        x = self.repeat_block(inputs=x,
                              stage="stage2",
                              repeat=self.repeat,
                              strides=(2, 2),
                              nf=self.nfilter[1])
        x = self.repeat_block(inputs=x,
                              stage="stage3",
                              repeat=self.repeat,
                              strides=(2, 2),
                              nf=self.nfilter[2])
        x = Activation("relu", name="end_act")(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)
        x = Flatten()(x)
        x = Dense(self.num_class,
                  activation="softmax",
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(delta))(x)
        model = Model(inputs, x)
        return model
