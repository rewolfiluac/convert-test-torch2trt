import random
from typing import Tuple, List

import cv2
import numpy as np


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def draw_bbox(
    img: np.ndarray,
    bbox: List[int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    text_scale: float = 1.0,
    thickness: int = 5,
) -> None:
    cv2.rectangle(
        img,
        pt1=(bbox[0], bbox[1]),
        pt2=(bbox[2], bbox[3]),
        color=color,
        thickness=thickness,
    )
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)

    # Prints the text.
    cv2.rectangle(img, (bbox[0], bbox[1] - h), (bbox[0] + w, bbox[1]), color, -1)
    cv2.putText(
        img,
        label,
        (bbox[0], bbox[1] - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        1,
    )
