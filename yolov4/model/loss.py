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
import numpy as np
import tensorflow as tf


class Loss(tf.keras.losses.Loss):
    def __init__(self, anchors: np.ndarray, xy_scales: np.ndarray, input_size: int, strides: np.ndarray, batch_size: int):
        super(Loss, self).__init__()

        self.input_size = input_size
        self.batch_index = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]

        self.anchor_grid_xywhs = []
        self.output_size_vector = []
        self.xy_scale_vector = []
        self.xy_grids = []

        output_sizes = self.input_size // strides

        for anchor, xy_scale, output_size in zip(anchors, xy_scales, output_sizes):
            vector_shape = (3 * output_size ** 2,)
            self.output_size_vector.append(np.full(vector_shape, output_size, np.float32))
            self.xy_scale_vector.append(np.full(vector_shape, xy_scale, np.float32))

            xy_grid = np.meshgrid(range(output_size), range(output_size))
            xy_grid = np.stack(xy_grid, -1)[:, :, np.newaxis]
            xy_grid = np.tile(xy_grid, [1, 1, 3, 1]).astype(np.float32)
            self.xy_grids.append(xy_grid.reshape((-1, 2)).astype(np.float32))

            xy = (xy_grid + 0.5 * (1 - xy_scale)) / output_size
            wh = np.tile(anchor / self.input_size, (output_size, output_size, 1, 1))
            self.anchor_grid_xywhs.append(np.concatenate([xy, wh], -1).reshape((-1, 4)))

        self.output_size_vector = tf.concat(self.output_size_vector, 0)[:, tf.newaxis]
        self.xy_scale_vector = tf.concat(self.xy_scale_vector, 0)[:, tf.newaxis]

        self.xy_grids = tf.concat(self.xy_grids, 0)

        self.anchor_grid_xywhs = tf.concat(self.anchor_grid_xywhs, 0)
        self.anchor_grid_xywhs = \
            tf.broadcast_to(self.anchor_grid_xywhs[tf.newaxis, ...], (batch_size, *tf.shape(self.anchor_grid_xywhs)))

    def call(self, y_true, y_pred):
        truth_xywhs = y_true[..., :4]
        truth_confidences = y_true[..., 4]
        truth_class_probs = y_true[..., 5:]

        predicted_xywhs = y_pred[..., :4]
        predicted_xy_probs = y_pred[..., 4:6]
        predicted_wh_exponents = y_pred[..., 6:8]
        predicted_confidences = y_pred[..., 8]
        predicted_class_probs = y_pred[..., 9:]

        grid_args_for_best_iou = tf.argmax(iou(truth_xywhs, self.anchor_grid_xywhs), -1, tf.int32)[..., tf.newaxis]
        grid_args_for_best_iou = tf.concat([
            tf.broadcast_to(self.batch_index, tf.shape(grid_args_for_best_iou)),
            grid_args_for_best_iou
        ], -1)

        output_sizes = tf.gather_nd(self.output_size_vector, grid_args_for_best_iou[..., 1:])
        xy_scales = tf.gather_nd(self.xy_scale_vector, grid_args_for_best_iou[..., 1:])
        xy_grids = tf.gather_nd(self.xy_grids[..., :2], grid_args_for_best_iou[..., 1:])
        truth_xy_probs = (truth_xywhs[..., :2] * output_sizes - xy_grids - 0.5) / xy_scales + 0.5
        predicted_xy_probs = tf.gather_nd(predicted_xy_probs, grid_args_for_best_iou)

        truth_wh_exponents = tf.math.log(
            truth_xywhs[..., 2:] / tf.gather_nd(self.anchor_grid_xywhs[..., 2:], grid_args_for_best_iou)
        )
        predicted_wh_exponents = tf.gather_nd(predicted_wh_exponents, grid_args_for_best_iou)

        confidences_mask = tf.tensor_scatter_nd_update(
            tf.cast(0.5 > tf.reduce_max(iou(predicted_xywhs, truth_xywhs), -1), tf.float32),
            grid_args_for_best_iou,
            tf.ones(tf.shape(grid_args_for_best_iou)[:-1])
        )
        truth_confidences = confidences_mask * tf.tensor_scatter_nd_update(
            tf.zeros(tf.shape(predicted_confidences)),
            grid_args_for_best_iou,
            truth_confidences
        )
        predicted_confidences = confidences_mask * predicted_confidences

        predicted_class_probs = tf.gather_nd(predicted_class_probs, grid_args_for_best_iou)

        loss_xys = tf.keras.backend.binary_crossentropy(truth_xy_probs, predicted_xy_probs)
        loss_whs = tf.math.squared_difference(truth_wh_exponents, predicted_wh_exponents)
        loss_confidences = tf.keras.backend.binary_crossentropy(truth_confidences, predicted_confidences)
        loss_class_probs = tf.keras.backend.binary_crossentropy(truth_class_probs, predicted_class_probs)

        return (tf.reduce_mean(tf.reduce_sum(loss_xys, axis=(1, 2)))
                + tf.reduce_mean(tf.reduce_sum(loss_whs, axis=(1, 2)))
                + tf.reduce_mean(tf.reduce_sum(loss_confidences, axis=(1,)))
                + tf.reduce_mean(tf.reduce_sum(loss_class_probs, axis=(1, 2))))


@tf.function
def iou(xywhs1, xywhs2):
    half_whs1 = xywhs1[..., 2:] / 2
    half_whs2 = xywhs2[..., 2:] / 2

    top_left1 = xywhs1[..., :2] - half_whs1
    top_left2 = xywhs2[..., :2] - half_whs2
    bottom_right1 = xywhs1[..., :2] + half_whs1
    bottom_right2 = xywhs2[..., :2] + half_whs2

    intersection_top_left = tf.maximum(top_left1[..., tf.newaxis, :], top_left2[..., tf.newaxis, :, :])
    intersection_bottom_right = tf.minimum(bottom_right1[..., tf.newaxis, :], bottom_right2[..., tf.newaxis, :, :])

    one = tf.math.reduce_prod(bottom_right1 - top_left1, -1)
    two = tf.math.reduce_prod(bottom_right2 - top_left2, -1)

    condition = tf.reduce_prod(tf.cast(intersection_top_left < intersection_bottom_right, tf.float32), -1)
    i = tf.math.reduce_prod(intersection_bottom_right - intersection_top_left, -1) * condition
    u = one[..., tf.newaxis] + two[..., tf.newaxis, :]
    u -= i

    return i / u
