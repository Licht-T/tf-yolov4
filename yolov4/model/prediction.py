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

import tensorflow as tf
from tensorflow.keras import Model, Sequential

from .convolution import Activation, Convolution
from .darknet import Darknet53
from .neck import SpatialPyramidPooling, PathAggregationNetworkSmallBoundingBox, \
    PathAggregationNetworkMediumBoundingBox, PathAggregationNetworkLargeBoundingBox


class Prediction(Model):
    NAME = 'Prediction'

    def __init__(self, num_classes: int, anchors, xy_scales, input_size):
        super(Prediction, self).__init__(name=self.NAME)
        self.num_classes = num_classes
        self.anchors = anchors
        self.xy_scales = xy_scales
        self.input_size = input_size

        self.darknet53 = Darknet53()

        self.spatial_pyramid_pooling = SpatialPyramidPooling()

        self.path_aggregation_network_small_bbox = PathAggregationNetworkSmallBoundingBox()
        self.sequential_small_bbox = Sequential()

        output_filters = 3 * (num_classes + 5)
        self.sequential_small_bbox.add(Convolution(256, 3, activation=Activation.LEAKY_RELU))
        self.sequential_small_bbox.add(Convolution(output_filters, 1, activation=None))

        self.path_aggregation_network_medium_bbox = PathAggregationNetworkMediumBoundingBox()
        self.sequential_medium_bbox = Sequential()

        self.sequential_medium_bbox.add(Convolution(512, 3, activation=Activation.LEAKY_RELU))
        self.sequential_medium_bbox.add(Convolution(output_filters, 1, activation=None))

        self.path_aggregation_network_large_bbox = PathAggregationNetworkLargeBoundingBox()
        self.sequential_large_bbox = Sequential()

        self.sequential_large_bbox.add(Convolution(1024, 3, activation=Activation.LEAKY_RELU))
        self.sequential_large_bbox.add(Convolution(output_filters, 1, activation=None))

    def call(self, x, **kwargs):
        residual1, residual2, residual3 = self.darknet53(x)

        pooled = self.spatial_pyramid_pooling(residual3)

        aggregated1, aggregated1_shortcut = self.path_aggregation_network_small_bbox([residual1, residual2, pooled])
        small_bbox = self.sequential_small_bbox(aggregated1)

        aggregated2 = self.path_aggregation_network_medium_bbox([aggregated1, aggregated1_shortcut])
        medium_bbox = self.sequential_medium_bbox(aggregated2)

        aggregated3 = self.path_aggregation_network_large_bbox([aggregated2, pooled])
        large_bbox = self.sequential_large_bbox(aggregated3)

        s = decode(small_bbox, self.num_classes, self.anchors[0], self.xy_scales[0], self.input_size)
        m = decode(medium_bbox, self.num_classes, self.anchors[1], self.xy_scales[1], self.input_size)
        l = decode(large_bbox, self.num_classes, self.anchors[2], self.xy_scales[2], self.input_size)

        return tf.concat((s, m, l), 1)

    def set_darknet_weights(self, fd: typing.BinaryIO) -> None:
        self.darknet53.set_darknet_weights(fd)
        self.spatial_pyramid_pooling.set_darknet_weights(fd)

        self.path_aggregation_network_small_bbox.set_darknet_weights(fd)
        self._set_sequential_darknet_weights(self.sequential_small_bbox, fd)

        self.path_aggregation_network_medium_bbox.set_darknet_weights(fd)
        self._set_sequential_darknet_weights(self.sequential_medium_bbox, fd)

        self.path_aggregation_network_large_bbox.set_darknet_weights(fd)
        self._set_sequential_darknet_weights(self.sequential_large_bbox, fd)

    @staticmethod
    def _set_sequential_darknet_weights(sequential: Sequential, fd: typing.BinaryIO) -> None:
        layer: Convolution
        for layer in sequential.layers:
            layer.set_darknet_weights(fd)


@tf.function
def decode(output, num_classes, anchor, xy_scale, input_size):
    batch_size = tf.shape(output)[0]
    output_size = tf.shape(output)[1]
    output = tf.reshape(output, (batch_size, output_size, output_size, 3, 5 + num_classes))

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, -1)[:, :, tf.newaxis]
    xy_grid = tf.tile(xy_grid[tf.newaxis, ...], [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    dxdy_activated = tf.sigmoid(output[:, :, :, :, :2])
    dwdh = output[:, :, :, :, 2:4]

    output = tf.concat(
        [
            (xy_scale * (dxdy_activated - 0.5) + xy_grid + 0.5) / tf.cast(output_size, tf.float32),
            (tf.exp(output[:, :, :, :, 2:4]) * anchor) / tf.cast(input_size, tf.float32),
            dxdy_activated,
            dwdh,
            tf.sigmoid(output[:, :, :, :, 4:]),
        ], -1
    )

    return tf.reshape(output, (batch_size, -1, 4 + 4 + 1 + num_classes))
