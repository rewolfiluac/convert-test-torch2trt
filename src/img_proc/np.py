from typing import Tuple, List

import cv2
import numpy as np

from img_proc.padding import calc_pad_size, pad


def preprocess(
    input_data: np.ndarray,
    resize_shape: Tuple[int, int],
    padding: bool = False,
    devide_max: bool = False,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    h, w = input_data.shape[:2]
    if padding and h != w:
        tblr = calc_pad_size(h, w)
        input_data = pad(input_data, tblr)
    input_data = cv2.resize(input_data, resize_shape)
    # imagenet color RGB, not BGR.
    input_data = input_data[:, :, ::-1]
    # (h, w, c) to (c, h, w)
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, 0)
    input_data = input_data.astype(np.float32)
    return normalize(input_data, devide_max, mean, std).ravel()


def normalize(
    input_data: np.ndarray,
    devide_max: bool,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    if devide_max:
        input_data = input_data / np.max(input_data)
    else:
        input_data = input_data / 255
    input_data[:, 0, :, :] = (input_data[:, 0, :, :] - mean[0]) / std[0]
    input_data[:, 1, :, :] = (input_data[:, 1, :, :] - mean[1]) / std[1]
    input_data[:, 2, :, :] = (input_data[:, 2, :, :] - mean[2]) / std[2]
    return input_data
