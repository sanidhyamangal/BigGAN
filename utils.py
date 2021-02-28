"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

from typing import Callable

import tensorflow as tf  # for deep learning


def orthogonal_initializer(scale) -> Callable:
    """
    Function for orthogoal initializer for conv layers
    """

    def orgtho_init(w:tf.Tensor) -> tf.Tensor:

        # reshaping image matrix to 2d for enforcing orthogonality
        _,_,_, c = w.shape.as_list()
        w = tf.reshape(w, [-1, c])

        # declare identity matrix
        identity = tf.eye(c)

        # perform wt*w
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)

        reg = tf.subtract(w_mul, identity)

        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return orgtho_init

def othrogonal_initializer_fc(scale) -> Callable:
    """
    orthogonal initializer for fully connected layers
    """

    def ortho_init_fc(w:tf.Tensor) -> tf.Tensor:

        # reshaping 2d tensor
        _,c = w.shape.as_list()

        # create an identity mapping
        identity = tf.eye(c)

        # perform orthogonal init
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        # loss for the layer
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss
    
    return ortho_init_fc
