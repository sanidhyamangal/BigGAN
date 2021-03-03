"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops
from functools import partial  # for creation of partial methods
from typing import Tuple  # for help with typings

import tensorflow as tf  # for deep learning based ops
from utils import orthogonal_initializer, othrogonal_initializer_fc


class BasicResBlock(tf.keras.layers.Layer):
    """
    Basic Block for residual networks
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int] = (3, 3),
                 padding: str = "same",
                 strides: Tuple[int] = (2, 2),
                 *args,
                 **kwargs):
        """
        Constructor of the model for creating a basic block for the residual network
        """
        super(BasicResBlock, self).__init__(*args, **kwargs)

        # first conv block
        self.conv1 = tf.keras.layers.Convolution2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=orthogonal_initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()

        # seconf conv block
        self.conv2 = tf.keras.layers.Convolution2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=(1, 1),
            kernel_initializer=orthogonal_initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # res conv block
        self.res_conv = tf.keras.layers.Convolution2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=strides,
            padding=padding,
            kernel_initializer=orthogonal_initializer)
        self.res_bn = tf.keras.layers.BatchNormalization()

        # final activation
        self.act2 = tf.keras.layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        training = kwargs.get('training', True)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        residual = self.res_conv(inputs)
        residual = self.res_bn(residual, training=training)

        output = self.act2(tf.add(x, residual))

        return output


class ResBottleNeckBlock(tf.keras.layers.Layer):
    """
    Bottle Neck based arch for higher order layers
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int] = (3, 3),
                 padding: str = "same",
                 strides: Tuple[int] = (2, 2),
                 *args,
                 **kwargs):
        super(ResBottleNeckBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        Conv2D = partial(tf.keras.layers.Convolution2D,
                         kernel_size=kernel_size,
                         filters=filters,
                         padding=padding,
                         strides=strides,
                         kernel_initializer=orthogonal_initializer)
        RelU = partial(tf.keras.layers.ReLU)
        BN = partial(tf.keras.layers.BatchNormalization)

        self.conv1 = Conv2D(kernel_size=(1, 1), strides=(1, 1))
        self.bn1 = BN()
        self.act1 = RelU()

        self.conv2 = Conv2D()
        self.bn2 = BN()
        self.act2 = RelU()

        self.conv3 = Conv2D(filters=filters * 4,
                            kernel_size=(1, 1),
                            strides=(1, 1))
        self.bn3 = BN()
        self.act3 = RelU()

        self.res_conv = Conv2D(filters=filters * 4, kernel_size=(1, 1))
        self.res_bn = BN()

        self.act4 = RelU()

    def call(self, inputs, *args, **kwargs):
        training = kwargs.pop('training', True)

        # first layer execution complete
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        # executing second layer
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        # executing third layer
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)

        # executing residual layer
        residual = self.res_conv(inputs)
        residual = self.res_bn(residual, training=training)

        return self.act3(tf.add(x, residual))
