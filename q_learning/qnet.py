import math
import os
import pickle
import numpy as np
import tensorflow as tf

L2_WEIGHT_DECAY = 0.001
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 0.001
LR = 0.0001


def basic_resblock(input_tensor, filters1, l2=L2_WEIGHT_DECAY):
    """
    (1) A convolution of 256 filters of kernel size 3 × 3 with stride 1
    (2) Batch normalization
    (3) A rectifier nonlinearity
    (4) A convolution of 256 filters of kernel size 3 × 3 with stride 1
    (5) Batch normalization
    (6) A skip connection that adds the input to the block
    (7) A rectifier nonlinearity
    """

    x = tf.keras.layers.Conv2D(filters1, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(input_tensor)

    x = tf.keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters1, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(input_tensor)

    x = tf.keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


class QResNet:
    def __init__(self):
        L2 = L2_WEIGHT_DECAY
        inputs = tf.keras.Input(shape=(10, 10, 7 * 14))
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(L2), name="c1")(inputs)

        x = tf.keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)
        x = basic_resblock(x, 256)

        x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1, 1), use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(L2), name="last")(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(L2), use_bias=False, name="d1")(x)
        x = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x)
        x = tf.keras.layers.Activation('relu')(x)
        y = tf.keras.layers.Dense(1656, use_bias=False, name="d2")(x)

        self.Model = tf.keras.Model(inputs=[inputs], outputs=[y])

    def compile(self, lr=LR):
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
        loss1 = tf.keras.losses.MeanSquaredError()

        self.Model.compile(optimizer=opt, loss=loss1)


drop_rate = 0
