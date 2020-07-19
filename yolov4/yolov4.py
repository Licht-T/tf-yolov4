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
        self.batchsize = 64

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

    def predict(self, frame: np.ndarray, show=False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_image = (image_util.resize_with_padding(rgb_frame, self.input_size) / 255)[np.newaxis, ...]

        output = self.model.predict(resized_image)
        boxes, _, confidences, class_probabilities = np.split(output, np.cumsum((4, 4, 1)), -1)
        boxes = boxes.reshape((-1, 4))
        confidences = confidences.reshape((-1,))
        class_probabilities = class_probabilities.reshape((-1, len(self.class_names)))

        height, width, _ = frame.shape

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

    def fit(self, x, y, epoch=1000):
        self.model.fit(x, y, self.batchsize, epoch)
