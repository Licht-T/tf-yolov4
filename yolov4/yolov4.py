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
import cv2
import numpy as np
import tensorflow as tf
import typing

from .model import prediction
from .model.loss import Loss
from .util import file, image_util


class YOLOv4:
    def __init__(self, class_names: typing.List[str]):
        self.anchors = np.array([
            [
                [12, 16],
                [19, 36],
                [40, 28]
            ], [
                [36, 75],
                [76, 55],
                [72, 146]
            ], [
                [142, 110],
                [192, 243],
                [459, 401],
            ]
        ], dtype=np.float32)
        self.input_size = 608
        self.strides = np.array([8, 16, 32])
        self.xy_scales = np.array([1.2, 1.1, 1.05])
        self.batchsize = 2
        self.steps_per_epoch = 100

        self.class_names = class_names

        tf.keras.backend.clear_session()
        input = tf.keras.layers.Input([self.input_size, self.input_size, 3])
        self.model = prediction.Prediction(len(self.class_names), self.anchors, self.xy_scales, self.input_size)
        self.model(input)

    def load_weights(self, weights_path: str) -> None:
        self.model.load_weights(weights_path).expect_partial()

    def load_darknet_weights(self, weights_file: str) -> None:
        with open(weights_file, 'rb') as fd:
            major, minor, revision = file.get_ndarray_from_fd(fd, dtype=np.int32, count=3)
            if major * 10 + minor >= 2:
                seen = file.get_ndarray_from_fd(fd, dtype=np.int64, count=1)[0]
            else:
                seen = file.get_ndarray_from_fd(fd, dtype=np.int32, count=1)[0]

            self.model.set_darknet_weights(fd)

    def predict(self, frame: typing.Union[np.ndarray, tf.Tensor], show: bool = False) \
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = tf.convert_to_tensor(frame)
        resized_image = image_util.preprocess(frame, self.input_size)[tf.newaxis, ...]

        output = self.model.predict(resized_image)
        boxes, _, confidences, class_probabilities = np.split(output, np.cumsum((4, 4, 1)), -1)
        boxes = boxes.reshape((-1, 4))
        confidences = confidences.reshape((-1,))
        class_probabilities = class_probabilities.reshape((-1, len(self.class_names)))

        height, width, _ = frame.get_shape().as_list()

        ratio = max(width, height) / min(width, height)
        i = 1 if width > height else 0
        boxes[:, i] = ratio * (boxes[:, i] - 0.5) + 0.5
        boxes[:, 3 if width > height else 2] *= ratio

        boxes[:, 2:] /= 2
        center_xy = boxes[:, :2].copy()
        boxes[:, :2] -= boxes[:, 2:]
        boxes[:, 2:] += center_xy

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height

        scores = confidences[:, np.newaxis] * class_probabilities

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes.reshape((1, -1, 1, 4)), scores[np.newaxis, ...],
            50, 50,
            0.5, 0.4,
            clip_boxes=False
        )
        boxes = boxes.numpy()[0]
        scores = scores.numpy()[0]
        classes = classes.numpy()[0].astype(np.int)

        if show:
            frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)

            window_name = 'result'
            output_frame = image_util.draw_bounding_boxes(frame, boxes, classes, scores, self.class_names)
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, output_frame)

            while cv2.waitKey(10) & 0xFF != ord('q'):
                pass
            cv2.destroyWindow(window_name)

        return boxes, classes, scores

    def compile(self):
        self.model.compile('Adam', Loss(self.anchors, self.xy_scales, self.input_size, self.strides, self.batchsize))

    def fit(self, image_paths: typing.List[str], label_paths: typing.List[str], epochs: int = 1000):
        image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_path_ds = tf.data.Dataset.from_tensor_slices(label_paths)

        image_label_path_ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

        image_label_ds = image_label_path_ds.map(
            lambda x, y: _load_and_preprocess_image_and_label(len(self.class_names), self.input_size, x, y),
            tf.data.experimental.AUTOTUNE
        ).repeat().batch(self.batchsize)

        self.model.fit(image_label_ds, epochs=epochs, steps_per_epoch=self.steps_per_epoch)


@tf.function
def _decode_csv(line: tf.Tensor):
    return tf.io.decode_csv(line, [0.0, 0.0, 0.0, 0.0, 0.0], ' ')


@tf.function
def _load_and_preprocess_image_and_label(num_classes: int, input_size: int, image_path: str, label_path: str) \
        -> typing.Tuple[tf.Tensor, tf.Tensor]:
    img = image_util.load_image(image_path)
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    labels = tf.zeros((0, 5))
    for e in tf.data.TextLineDataset(label_path).map(_decode_csv):
        x = tf.stack(e)[tf.newaxis, :]
        labels = tf.concat([labels, x], 0)

    n_labels = tf.shape(labels)[0]

    label_index = tf.range(n_labels)[:, tf.newaxis]
    labels_converted = tf.zeros((n_labels, 5 + num_classes), np.float32)
    ones = tf.ones((n_labels,))

    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, tf.broadcast_to(4, (n_labels, 1))], -1),
        ones
    )
    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, (5 + tf.cast(labels[:, 0], tf.int32))[:, tf.newaxis]], -1),
        ones
    )

    bboxes = labels[:, 1:]
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]

    ratio = tf.cast(tf.maximum(width, height) / tf.minimum(width, height), tf.float32)
    if width > height:
        y = (y - 0.5) / ratio + 0.5
        h /= ratio
    else:
        x = (x - 0.5) / ratio + 0.5
        w /= ratio

    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, tf.broadcast_to(0, (n_labels, 1))], -1),
        x
    )
    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, tf.broadcast_to(1, (n_labels, 1))], -1),
        y
    )
    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, tf.broadcast_to(2, (n_labels, 1))], -1),
        w
    )
    labels_converted = tf.tensor_scatter_nd_update(
        labels_converted,
        tf.concat([label_index, tf.broadcast_to(3, (n_labels, 1))], -1),
        h
    )

    return image_util.preprocess(img, input_size), labels_converted
