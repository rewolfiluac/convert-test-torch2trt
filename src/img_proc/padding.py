from typing import List, Tuple

import cv2
import numpy as np


def calc_pad_size(
    h: int,
    w: int,
) -> Tuple[int, int, int, int]:
    space = abs(h - w)
    if h > w:
        tblr = (0, 0, space // 2, space - space // 2)
    else:
        tblr = (space // 2, space - space // 2, 0, 0)
    return tblr


def pad(
    input_data: np.ndarray,
    tblr: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    return cv2.copyMakeBorder(
        input_data,
        top=tblr[0],
        bottom=tblr[1],
        left=tblr[2],
        right=tblr[3],
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
