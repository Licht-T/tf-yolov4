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
import typing

import tensorflow as tf


@tf.function
def preprocess(image: typing.Union[np.ndarray, tf.Tensor], size: int) -> tf.Tensor:
    return tf.image.resize_with_pad(image, size, size) / 255.0


@tf.function
def load_image(path: str) -> tf.Tensor:
    return tf.image.decode_image(tf.io.read_file(path), 3)


def draw_bounding_boxes(image: np.ndarray, bboxes: np.ndarray, classes: np.ndarray,
                        scores: np.ndarray, num_classes: int) -> np.ndarray:
    image = np.copy(image)

    colors = [tuple(xx) for xx in cv2.applyColorMap(
        np.array([int(255.0 * x / num_classes) for x in range(num_classes)], dtype=np.uint8).reshape((1, -1)),
        cv2.COLORMAP_RAINBOW
    ).reshape((-1, 3)).tolist()]

    for bbox, cls, score in zip(bboxes, classes, scores):
        color = colors[cls]
        cv2.rectangle(image, tuple(bbox[:2].astype(np.int)), tuple(bbox[2:].astype(np.int)), color, 2)

    return image
