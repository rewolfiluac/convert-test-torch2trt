from typing import Tuple, List
import time
import logging

import cv2
import numpy as np

from img_proc.padding import calc_pad_size, pad
from preproc import (
    padToSquare,
    normalize,
)


def preprocess(
    input_data: np.ndarray,
    resize_shape: Tuple[int, int],
    padding: bool = False,
    devide_max: bool = False,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    if padding and input_data.shape[0] != input_data.shape[1]:
        input_data = scale_box(input_data, *resize_shape)
        h, w, c = input_data.shape
        out_data = np.empty(
            (h if h > w else w, h if h > w else w, 3),
            dtype=np.uint8,
        )
        padToSquare(
            input_data,
            out_data,
        )
        input_data = out_data
    else:
        input_data = cv2.resize(input_data, resize_shape)

    h, w, c = input_data.shape

    # BGR2RGB
    input_data = input_data[:, :, ::-1]

    ret_data = np.empty((h * w * c), dtype=np.float32)
    normalize(
        input_data,
        ret_data,
        mean,
        std,
        float(np.max(input_data)) if devide_max else 255.0,
    )
    return ret_data


def scale_box(
    img: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)
    return cv2.resize(img, dsize=(nw, nh))
