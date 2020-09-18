"""
MIT License

Copyright (c) 2020 Licht Takeuchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Union
from enum import Enum

import functools
import typing

import numpy as np

import tensorflow as tf

tf.random.Generator = None  # Patch for a bug
import tensorflow_addons as tfa
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Layer

from ..util import file


class Activation(Enum):
    RELU = layers.ReLU
    LEAKY_RELU = functools.partial(layers.LeakyReLU, 0.1)
    MISH = functools.partial(layers.Activation, tfa.activations.mish)


class Convolution(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int = 1,
                 activation: Union[Activation, None] = Activation.MISH, **kwargs):
        super(Convolution, self).__init__(**kwargs)
        self.filters = filters
        self.input_dim = None
        self.kernel_size = kernel_size
        strides = strides

        self.sequential = Sequential()
        self._batch_normalization = None

        if strides == 2:
            self.sequential.add(layers.ZeroPadding2D(((1, 0), (1, 0))))

        padding = 'same' if strides == 1 else 'valid'
        use_bias = activation is None
        self._conv = layers.Conv2D(
            self.filters, self.kernel_size, strides,
            padding=padding, use_bias=use_bias,
            kernel_initializer='he_normal',
        )

        self.sequential.add(self._conv)

        if activation is not None:
            self._batch_normalization = layers.BatchNormalization(epsilon=1e-5)
            self.sequential.add(self._batch_normalization)
            self.sequential.add(activation.value())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x, **kwargs):
        return self.sequential(x)

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        conv_bias = None

        if self._batch_normalization is not None:
            beta = file.get_ndarray_from_fd(fd, np.float32, self.filters)
            gamma = file.get_ndarray_from_fd(fd, np.float32, self.filters)
            mean = file.get_ndarray_from_fd(fd, np.float32, self.filters)
            variance = file.get_ndarray_from_fd(fd, np.float32, self.filters)
            self._batch_normalization.set_weights([gamma, beta, mean, variance])
        else:
            conv_bias = file.get_ndarray_from_fd(fd, np.float32, self.filters)

        shape = (self.filters, self.input_dim, self.kernel_size, self.kernel_size)
        conv_coefficients = file.get_ndarray_from_fd(fd, np.float32, np.product(shape)).reshape(shape)
        conv_coefficients = conv_coefficients.transpose([2, 3, 1, 0])

        weights = [conv_coefficients]
        weights.append(conv_bias) if conv_bias is not None else None
        self._conv.set_weights(weights)
