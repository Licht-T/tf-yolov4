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
from tensorflow.keras import Model, Sequential

from .convolution import Activation, Convolution
from .residual import ResNetWithCrossStagePartialConnection


class Darknet53(Model):
    NAME = 'Darknet53'

    def __init__(self):
        super(Darknet53, self).__init__(name=self.NAME)
        self.conv = Convolution(32, 3)

        self.res1 = ResNetWithCrossStagePartialConnection(64, 64, 1)
        self.res2 = ResNetWithCrossStagePartialConnection(128, 64, 2)
        self.res3 = ResNetWithCrossStagePartialConnection(256, 128, 8)
        self.res4 = ResNetWithCrossStagePartialConnection(512, 256, 8)
        self.res5 = ResNetWithCrossStagePartialConnection(1024, 512, 4)

        self.residual_layers = [self.res1, self.res2, self.res3, self.res4, self.res5]

        self.conv_sequential = Sequential()
        small = [512, 1]
        large = [1024, 3]

        for filters, kernel_size in [small, large, small]:
            self.conv_sequential.add(Convolution(filters, kernel_size, activation=Activation.LEAKY_RELU))

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        residual1 = self.res3(x)

        residual2 = self.res4(residual1)

        x = self.res5(residual2)
        residual3 = self.conv_sequential(x)

        return [residual1, residual2, residual3]

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        self.conv.set_darknet_weights(fd)

        layer_res: ResNetWithCrossStagePartialConnection
        for layer_res in self.residual_layers:
            layer_res.set_darknet_weights(fd)

        layer: Convolution
        for layer in self.conv_sequential.layers:
            layer.set_darknet_weights(fd)
