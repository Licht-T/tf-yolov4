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

from tensorflow.keras import Model, layers, Sequential

from .convolution import Convolution


class ResidualConvolution(Model):
    def __init__(self, filters1: int, filters2: int):
        super(ResidualConvolution, self).__init__()
        self.sequential = Sequential()
        self.sequential.add(Convolution(filters1, 1))
        self.sequential.add(Convolution(filters2, 3))

        self.add = layers.Add()

    def call(self, x, **kwargs):
        return self.add([x, self.sequential(x)])

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        layer: Convolution
        for layer in self.sequential.layers:
            layer.set_darknet_weights(fd)


class ResidualConvolutionLoop(Model):
    def __init__(self, filters1: int, filters2: int, loop: int):
        super(ResidualConvolutionLoop, self).__init__()
        self.sequential = Sequential()
        self.loop = loop

        for _ in range(loop):
            self.sequential.add(ResidualConvolution(filters1, filters2))

    def call(self, x, **kwargs):
        return self.sequential(x)

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        layer: ResidualConvolution
        for layer in self.sequential.layers:
            layer.set_darknet_weights(fd)


class ResNetWithCrossStagePartialConnection(Model):
    def __init__(self, filters1: int, filters2: int, iteration: int):
        super(ResNetWithCrossStagePartialConnection, self).__init__()
        self.pre_conv = Convolution(filters1, 3, 2)

        self.conv_before = Convolution(filters2, 1)
        self.residual_convolution_loop = ResidualConvolutionLoop(filters1 // 2, filters2, iteration)
        self.conv_after = Convolution(filters2, 1)

        self.shortcut_conv = Convolution(filters2, 1)

        self.concat = layers.Concatenate(-1)

        self.post_conv = Convolution(filters1, 1)

    def call(self, x, **kwargs):
        x = self.pre_conv(x)

        y = self.conv_before(x)
        y = self.residual_convolution_loop(y)
        y = self.conv_after(y)

        x = self.concat([y, self.shortcut_conv(x)])

        return self.post_conv(x)

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        self.pre_conv.set_darknet_weights(fd)
        self.shortcut_conv.set_darknet_weights(fd)
        self.conv_before.set_darknet_weights(fd)

        self.residual_convolution_loop.set_darknet_weights(fd)

        self.conv_after.set_darknet_weights(fd)
        self.post_conv.set_darknet_weights(fd)
