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
        self.xy_scales = xy_scales
        self.strides = strides
        self.output_sizes = self.input_size // self.strides
        self.batch_size = batch_size
        self.batch_index = tf.range(batch_size)[:, tf.newaxis]
        self.num_anchor_types = anchors.shape[1]

        tmp = np.pad(self.output_sizes, (1, 0)).cumsum()[:, np.newaxis]
        self.output_ranges = np.concatenate([tmp[:-1, :], tmp[1:, :]], axis=1)

        self.anchor_grid_xywhs_list = []

        for anchor, xy_scale, output_size in zip(anchors, self.xy_scales, self.output_sizes):
            xy_grid = np.meshgrid(range(output_size), range(output_size))
            xy_grid = np.stack(xy_grid, -1)[:, :, np.newaxis]
            xy_grid = np.tile(xy_grid, [1, 1, 3, 1]).astype(np.float32)

            xy = (xy_grid + 0.5 * (1 - xy_scale)) / output_size
            wh = np.tile(anchor / self.input_size, (output_size, output_size, 1, 1))

            xywh = tf.convert_to_tensor(np.concatenate([xy, wh], -1).reshape((-1, 4)))

            self.anchor_grid_xywhs_list.append(tf.broadcast_to(xywh, (self.batch_size, *tf.shape(xywh))))

    def call(self, y_true, y_pred):
        split_sizes = self.num_anchor_types * (self.output_sizes ** 2)

        predicted_xywhs_sml = tf.split(y_pred[:, :, :4], split_sizes, 1)
        predicted_xy_probs_sml = tf.split(y_pred[:, :, 4:6], split_sizes, 1)
        predicted_wh_exponents_sml = tf.split(y_pred[:, :, 6:8], split_sizes, 1)
        predicted_confidences_sml = tf.split(y_pred[:, :, 8], split_sizes, 1)
        predicted_class_probs_sml = tf.split(y_pred[:, :, 9:], split_sizes, 1)

        truth_xywhs = y_true[:, :, :4]
        truth_confidences = y_true[:, :, 4]
        truth_class_probs = y_true[:, :, 5:]

        loss_xys = 0.
        loss_whs = 0.
        loss_confidences = 0.
        loss_probabilities = 0.

        for i, output_size in enumerate(self.output_sizes):
            predicted_xywhs = predicted_xywhs_sml[i]
            predicted_xy_probs = predicted_xy_probs_sml[i]
            predicted_wh_exponents = predicted_wh_exponents_sml[i]
            predicted_confidences = predicted_confidences_sml[i]
            predicted_class_probs = predicted_class_probs_sml[i]
            anchor_grid_xywhs = self.anchor_grid_xywhs_list[i]
            xy_scale = self.xy_scales[i]

            anchor_grid_args_for_best_iou_to_truth = tf.argmax(iou(truth_xywhs, anchor_grid_xywhs), 2, tf.int32)
            anchor_grid_args_for_best_iou_to_truth = anchor_grid_args_for_best_iou_to_truth[..., tf.newaxis]
            anchor_grid_args_for_best_iou_to_truth = tf.concat([
                tf.broadcast_to(self.batch_index[:, :, tf.newaxis], tf.shape(anchor_grid_args_for_best_iou_to_truth)),
                anchor_grid_args_for_best_iou_to_truth
            ], -1)

            iou_over_predicted_and_truth = iou(predicted_xywhs, truth_xywhs)
            truth_args_for_best_iou_to_predicted = tf.argmax(iou_over_predicted_and_truth, 2, tf.int32)
            truth_args_for_best_iou_to_predicted = truth_args_for_best_iou_to_predicted[..., tf.newaxis]
            truth_args_for_best_iou_to_predicted = tf.concat([
                tf.broadcast_to(self.batch_index[:, :, tf.newaxis], tf.shape(truth_args_for_best_iou_to_predicted)),
                truth_args_for_best_iou_to_predicted
            ], -1)
            confidences_mask = tf.cast(0.7 > tf.reduce_max(iou_over_predicted_and_truth, 2), tf.float32)
            confidences_mask = tf.tensor_scatter_nd_update(
                confidences_mask,
                anchor_grid_args_for_best_iou_to_truth,
                tf.ones(tf.shape(anchor_grid_args_for_best_iou_to_truth)[:-1])
            )

            truth_xy_probs = \
                truth_xywhs[:, :, :2] - tf.gather_nd(predicted_xywhs, anchor_grid_args_for_best_iou_to_truth)[:, :, :2]
            truth_xy_probs *= tf.cast(output_size, tf.float32)
            truth_xy_probs -= 0.5
            truth_xy_probs /= xy_scale
            truth_xy_probs += 0.5
            predicted_xy_probs = tf.gather_nd(predicted_xy_probs, anchor_grid_args_for_best_iou_to_truth)

            truth_whs = tf.math.log(
                truth_xywhs[:, :, 2:] * self.input_size
                / tf.gather_nd(anchor_grid_xywhs[:, :, 2:], anchor_grid_args_for_best_iou_to_truth)
                + 1e-16
            )
            predicted_whs = tf.gather_nd(predicted_wh_exponents, anchor_grid_args_for_best_iou_to_truth)

            truth_confidences = tf.gather_nd(truth_confidences, truth_args_for_best_iou_to_predicted) * confidences_mask
            predicted_confidences = predicted_confidences * confidences_mask

            predicted_class_probs = tf.gather_nd(predicted_class_probs, anchor_grid_args_for_best_iou_to_truth)

            loss_xys += tf.keras.losses.binary_crossentropy(truth_xy_probs, predicted_xy_probs)
            loss_whs += tf.reduce_sum(tf.keras.losses.mse(truth_whs, predicted_whs), 1)
            loss_confidences += tf.keras.losses.binary_crossentropy(truth_confidences, predicted_confidences)
            loss_probabilities += tf.keras.losses.binary_crossentropy(truth_class_probs, predicted_class_probs)

        loss = loss_xys + loss_whs / 2 + loss_confidences + loss_probabilities

        return loss


def iou(xywhs1, xywhs2):
    half_whs1 = xywhs1[:, :, 2:] / 2
    half_whs2 = xywhs2[:, :, 2:] / 2

    top_left1 = xywhs1[:, :, :2] - half_whs1
    top_left2 = xywhs2[:, :, :2] - half_whs2
    bottom_right1 = xywhs1[:, :, :2] + half_whs1
    bottom_right2 = xywhs2[:, :, :2] + half_whs2

    intersection_top_left = tf.maximum(top_left1[:, :, tf.newaxis, :], top_left2[:, tf.newaxis, :, :])
    intersection_bottom_right = tf.minimum(bottom_right1[:, :, tf.newaxis, :], bottom_right2[:, tf.newaxis, :, :])

    one = tf.math.reduce_prod(bottom_right1 - top_left1, 2)
    two = tf.math.reduce_prod(bottom_right2 - top_left2, 2)

    condition = tf.reduce_prod(tf.cast(intersection_top_left < intersection_bottom_right, tf.float32), 3)
    i = tf.math.reduce_prod(intersection_bottom_right - intersection_top_left, 3) * condition
    u = one[:, :, tf.newaxis] + two[:, tf.newaxis, :]
    u -= i

    return i / u
