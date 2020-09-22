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
import PIL.Image
import numpy as np
import tensorflow as tf
import typing

from .model import prediction
from .model.loss import Loss
from .util import file, image_util

VERSION = 'v1.0.4'


class YOLOv4:
    def __init__(self, num_classes: int = 80):
        """
        Constructor.
        :param num_classes: # of classes to be predicted. The default value is 80 (MS-COCO).
        """
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
        self.learning_rate = 0.001
        self.burn_in_steps = 1000

        self.num_classes = num_classes

        self.model = None

        self.init_model()

    def init_model(self) -> None:
        """
        Generate the Keras model. Automatically called in the constructor.
        Call this and re-generate the model after a model initialization parameter has changed.
        :return: None
        """
        self.model = prediction.Prediction(self.num_classes, self.anchors, self.xy_scales, self.input_size)
        self.model(tf.keras.layers.Input([self.input_size, self.input_size, 3]))

    def load_weights(self, weights_path: str = None) -> None:
        """
        Load a Keras weight file.
        :param weights_path: Weight file path. The default is None.
               If weights_path == None and self.num_classes == 80: Pre-trained COCO model is retrieved.
               If weights_path == None and self.num_classes != 80: Pre-trained backbone and SPP model is retrieved.
               Otherwise: weights_path file is loaded.
        :return: None
        """
        if weights_path is None:
            base_url = f'https://github.com/Licht-T/tf-yolov4/releases/download/{VERSION}'
            if self.num_classes == 80:
                weights_path = tf.keras.utils.get_file(
                    f'yolov4_pretrained_coco_{VERSION}.h5',
                    f'{base_url}/yolov4_pretrained_coco.h5',
                    cache_subdir='tf-yolov4'
                )
            else:
                weights_path = tf.keras.utils.get_file(
                    f'yolov4_pretrained_backbone_and_spp_only_{VERSION}.h5',
                    f'{base_url}/yolov4_pretrained_backbone_and_spp_only.h5',
                    cache_subdir='tf-yolov4'
                )

        self.model.load_weights(weights_path)

    def load_darknet_weights(self, weights_file: str) -> None:
        """
        Load a Darknet weight file.
        :param weights_path: Weight file path.
        :return: None
        """
        with open(weights_file, 'rb') as fd:
            major, minor, revision = file.get_ndarray_from_fd(fd, dtype=np.int32, count=3)
            if major * 10 + minor >= 2:
                seen = file.get_ndarray_from_fd(fd, dtype=np.int64, count=1)[0]
            else:
                seen = file.get_ndarray_from_fd(fd, dtype=np.int32, count=1)[0]

            self.model.set_darknet_weights(fd)

    def save_weights(self, path: str) -> None:
        """
        Save the current weight.
        :param path: Weight path to be saved.
        :return: None
        """
        self.model.save_weights(path)

    def predict(self, frame: typing.Union[np.ndarray, tf.Tensor], debug: bool = False) \
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Image prediction method.
        :param frame: (H, W, 3)-shaped RGB-image to be predicted (uint8 np.ndarray or tf.Tensor).
        :param debug: If True, generates the image with predicted bounding-boxes into the current directory.
        :return: A tuple of predicted bounding-boxes, its class IDs and its score.
        """
        frame = tf.convert_to_tensor(frame)
        resized_image = image_util.preprocess(frame, self.input_size)[tf.newaxis, ...]

        output = self.model.predict(resized_image)
        boxes, _, confidences, class_probabilities = np.split(output, np.cumsum((4, 4, 1)), -1)
        boxes = boxes.reshape((-1, 4))
        confidences = confidences.reshape((-1,))
        class_probabilities = class_probabilities.reshape((-1, self.num_classes))

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
        cond = np.where(boxes[:, 2] * boxes[:, 3] > .0)

        boxes = boxes[cond]
        scores = scores.numpy()[0][cond]
        classes = classes.numpy()[0].astype(np.int)[cond]

        if debug:
            im = PIL.Image.fromarray(frame.numpy())
            im = image_util.draw_bounding_boxes(im, boxes, classes, scores)
            im.save('./output.png')

        return boxes, classes, scores

    def compile(self) -> None:
        """
        Compile the model for training.
        :return: None
        """
        self.model.compile('Adam', Loss(self.anchors, self.xy_scales, self.input_size, self.strides, self.batchsize))

    def fit(
            self, image_paths: typing.List[str], label_paths: typing.List[str], epochs: int = 1000,
            callbacks: typing.List[tf.keras.callbacks.Callback] = None
    ) -> tf.keras.callbacks.History:
        """
        :param image_paths: List of training image file paths.
               image_paths[i] should be the image file path for the label text path label_paths[i].
        :param label_paths: List of training label text file paths.
               label_paths[i] should be the label text file path for the image file path image_paths[i].
               Each label text file has the Darknet style annotations.
        :param epochs: # of training epochs.
        :param callbacks: List of tf.keras.callbacks.Callback for training.
        :return: tf.keras.callbacks.History object.
        """
        if callbacks is None:
            callbacks = []

        image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_path_ds = tf.data.Dataset.from_tensor_slices(label_paths)

        image_label_path_ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

        image_label_ds = image_label_path_ds.map(
            lambda x, y: _load_and_preprocess_image_and_label(self.num_classes, self.input_size, x, y),
            tf.data.experimental.AUTOTUNE
        ).shuffle(10, reshuffle_each_iteration=True).repeat().batch(self.batchsize)

        return self.model.fit(
            image_label_ds, epochs=epochs, steps_per_epoch=self.steps_per_epoch,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(
                    lambda e, lr: _get_current_learning_rate(
                        e, self.learning_rate, self.burn_in_steps, self.steps_per_epoch
                    )
                )
            ] + callbacks,
        )


def _get_current_learning_rate(current_epoch, max_lr, burn_in_steps, steps_per_epoch):
    lr = max_lr * tf.math.pow(tf.math.minimum(1.0, (current_epoch + 1)/(burn_in_steps / steps_per_epoch)), 4)
    tf.print(tf.strings.format('Current learning rate: {}', (lr,)))
    return lr


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

    label_indices = tf.range(n_labels)[:, tf.newaxis]
    ones = tf.ones((n_labels,))

    labels_conf_and_probs = tf.tensor_scatter_nd_update(
        tf.zeros((n_labels, 1 + num_classes), np.float32),
        tf.concat([label_indices, tf.broadcast_to(0, (n_labels, 1))], -1),
        ones
    )
    labels_conf_and_probs = tf.tensor_scatter_nd_update(
        labels_conf_and_probs,
        tf.concat([label_indices, (1 + tf.cast(labels[:, 0:1], tf.int32))], -1),
        ones
    )

    bboxes = labels[:, 1:]
    x = bboxes[:, 0:1]
    y = bboxes[:, 1:2]
    w = bboxes[:, 2:3]
    h = bboxes[:, 3:4]

    ratio = tf.cast(tf.maximum(width, height) / tf.minimum(width, height), tf.float32)
    if width > height:
        y = (y - 0.5) / ratio + 0.5
        h /= ratio
    else:
        x = (x - 0.5) / ratio + 0.5
        w /= ratio

    return image_util.preprocess(img, input_size), tf.concat([x, y, w, h, labels_conf_and_probs], -1)
