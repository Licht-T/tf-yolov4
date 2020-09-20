# tf-yolov4
[![PyPI version](https://badge.fury.io/py/tf-yolov4.svg)](https://badge.fury.io/py/tf-yolov4)
![Upload Python Package](https://github.com/Licht-T/tf-yolov4/workflows/Upload%20Python%20Package/badge.svg)

[YOLOv4](https://arxiv.org/abs/2004.10934) implementation with Tensorflow 2.

## Install
```bash
pip instal tf-yolov4
```

## Example
### Prediction
```python
import numpy as np
import PIL.Image
import yolov4

# Default: num_classes=80
yo = yolov4.YOLOv4(num_classes=80)

# Default: weights_path=None
# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
# num_classes!=80 and weights_path=None: Pre-trained backbone and SPP model will be loaded.
# Otherwise: User-defined weight file will be loaded.
yo.load_weights(weights_path=None)

img = np.array(PIL.Image.open('./data/sf.jpg'))

# The image with predicted bounding-boxes is created if `debug=True`
boxes, classes, scores = yo.predict(img, debug=True)
```
![output](https://raw.githubusercontent.com/Licht-T/tf-yolov4/master/data/output.png)

### Load Darknet weight
```python
import yolov4

yo = yolov4.YOLOv4(num_classes=10)
yo.load_darknet_weights('/path/to/darknet_weight')
```

## TODO
* [x] Prediction
* [x] Load Darknet weight file
* [x] Pre-trained model
* [x] Basic training function and Loss definition
* [ ] Label-smoothed BCE loss
* [ ] c-IoU loss
* [ ] Training data augmentation
