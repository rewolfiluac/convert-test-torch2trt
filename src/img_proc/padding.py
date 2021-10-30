from typing import List, Tuple
from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass
class PaddingSize:
    top: int
    bottom: int
    left: int
    right: int


def calc_pad_size(
    h: int,
    w: int,
) -> PaddingSize:
    space = abs(h - w)
    if h > w:
        tblr = PaddingSize(0, 0, space // 2, space - space // 2)
    else:
        tblr = PaddingSize(space // 2, space - space // 2, 0, 0)
    return tblr


def pad(
    input_data: np.ndarray,
    tblr: PaddingSize,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    return cv2.copyMakeBorder(
        input_data,
        top=tblr.top,
        bottom=tblr.bottom,
        left=tblr.left,
        right=tblr.right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
