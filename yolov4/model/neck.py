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
import typing
from tensorflow.keras import layers, Model, Sequential

from .convolution import Activation, Convolution


class SpatialPyramidPooling(Model):
    NAME = 'SpatialPyramidPooling'

    def __init__(self):
        super(SpatialPyramidPooling, self).__init__(name=self.NAME)
        self.sequential = Sequential()

        padding = 'same'
        self.large_pool = layers.MaxPooling2D((13, 13), 1, padding)
        self.medium_pool = layers.MaxPooling2D((9, 9), 1, padding)
        self.small_pool = layers.MaxPooling2D((5, 5), 1, padding)
        self.concat = layers.Concatenate(-1)

        small = [512, 1]
        large = [1024, 3]

        for filters, kernel_size in [small, large, small]:
            self.sequential.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

    def call(self, x, **kwargs):
        return self.sequential(self.concat([self.large_pool(x), self.medium_pool(x), self.small_pool(x), x]))

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        layer: Convolution
        for layer in self.sequential.layers:
            layer.set_darknet_weights(fd)


class PathAggregationNetworkSmallBoundingBox(Model):
    NAME = 'PathAggregationNetworkSmallBoundingBox'

    def __init__(self):
        super(PathAggregationNetworkSmallBoundingBox, self).__init__(name=self.NAME)

        self.sequential1 = Sequential()
        self.layers_for_weight_load = []

        self.conv_residual3 = Convolution(256, 1, activation=Activation.LEAKY_RELU)
        self.layers_for_weight_load.append(self.conv_residual3)
        self.up_sampling3 = layers.UpSampling2D(interpolation='bilinear')
        self.conv_residual2 = Convolution(256, 1, activation=Activation.LEAKY_RELU)
        self.layers_for_weight_load.append(self.conv_residual2)
        self.concat23 = layers.Concatenate(-1)

        small1 = [256, 1]
        large1 = [512, 3]

        for filters, kernel_size in [small1, large1, small1, large1, small1]:
            self.sequential1.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

        self.layers_for_weight_load += self.sequential1.layers

        self.sequential2 = Sequential()

        self.conv_residual23 = Convolution(128, 1, activation=Activation.LEAKY_RELU)
        self.layers_for_weight_load.append(self.conv_residual23)
        self.up_sampling23 = layers.UpSampling2D(interpolation='bilinear')
        self.conv_residual1 = Convolution(128, 1, activation=Activation.LEAKY_RELU)
        self.layers_for_weight_load.append(self.conv_residual1)
        self.concat123 = layers.Concatenate(-1)

        small2 = [128, 1]
        large2 = [256, 3]

        for filters, kernel_size in [small2, large2, small2, large2, small2]:
            self.sequential2.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

        self.layers_for_weight_load += self.sequential2.layers

    def call(self, x, **kwargs):
        residual1, residual2, residual3 = x
        concat_residual23 = self.concat23([
            self.conv_residual2(residual2),
            self.up_sampling3(self.conv_residual3(residual3))
        ])
        convoluted_residual23 = self.sequential1(concat_residual23)

        concat_residual123 = self.concat123([
            self.conv_residual1(residual1),
            self.up_sampling23(self.conv_residual23(convoluted_residual23))
        ])

        return [self.sequential2(concat_residual123), convoluted_residual23]

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        layer: Convolution
        for layer in self.layers_for_weight_load:
            layer.set_darknet_weights(fd)


class PathAggregationNetworkMediumBoundingBox(Model):
    NAME = 'PathAggregationNetworkMediumBoundingBox'

    def __init__(self):
        super(PathAggregationNetworkMediumBoundingBox, self).__init__(name=self.NAME)

        self.sequential = Sequential()

        self.conv = Convolution(256, 3, 2, Activation.LEAKY_RELU)
        self.concat = layers.Concatenate(-1)

        small = [256, 1]
        large = [512, 3]

        for filters, kernel_size in [small, large, small, large, small]:
            self.sequential.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

    def call(self, x, **kwargs):
        aggregated_input, shortcut_input = x

        return self.sequential(self.concat([self.conv(aggregated_input), shortcut_input]))

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        self.conv.set_darknet_weights(fd)

        layer: Convolution
        for layer in self.sequential.layers:
            layer.set_darknet_weights(fd)


class PathAggregationNetworkLargeBoundingBox(Model):
    NAME = 'PathAggregationNetworkLargeBoundingBox'

    def __init__(self):
        super(PathAggregationNetworkLargeBoundingBox, self).__init__(name=self.NAME)

        self.sequential = Sequential()

        self.conv = Convolution(512, 3, 2, Activation.LEAKY_RELU)
        self.concat = layers.Concatenate(axis=-1)

        small = [512, 1]
        large = [1024, 3]

        for filters, kernel_size in [small, large, small, large, small]:
            self.sequential.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

    def call(self, x, **kwargs):
        aggregated_input, shortcut_input = x

        return self.sequential(self.concat([self.conv(aggregated_input), shortcut_input]))

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        self.conv.set_darknet_weights(fd)

        layer: Convolution
        for layer in self.sequential.layers:
            layer.set_darknet_weights(fd)
