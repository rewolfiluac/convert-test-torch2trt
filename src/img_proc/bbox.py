from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import cv2

from img_proc.padding import calc_pad_size, pad


BASE_IMG_SIZE = 300


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


def calc_line_size(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    img_size = h if h > w else w
    return img_size // BASE_IMG_SIZE


def calc_text_scale(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    img_size = h if h > w else w
    return img_size / BASE_IMG_SIZE / 4


def correct_bbox(
    img: np.ndarray,
    boxes: np.ndarray,
    input_height: int,
    input_width: int,
) -> List[BBox]:
    h, w = img.shape[:2]
    # calc padding size
    tblr = calc_pad_size(*img.shape[:2])
    paded_h = h + tblr.top + tblr.bottom
    paded_w = w + tblr.left + tblr.right

    # correct bbox for padding and resize
    h_ratio = paded_h / input_height
    w_ratio = paded_w / input_width
    return [
        BBox(
            x1=int(box[0] * w_ratio - tblr.left),
            y1=int(box[1] * h_ratio - tblr.top),
            x2=int(box[2] * w_ratio - tblr.left),
            y2=int(box[3] * h_ratio - tblr.top),
        )
        for box in boxes
    ]


def draw_bboxs(
    img: np.ndarray,
    classes_dict: Dict[int, str],
    bboxes: List[BBox],
    out_classes: List[int],
):
    text_scale = calc_text_scale(img)
    thickness = calc_line_size(img)
    for box, cls_idx in zip(bboxes, out_classes):
        draw_bbox(
            img,
            box,
            classes_dict[cls_idx],
            text_scale=text_scale,
            thickness=thickness,
        )


def draw_bbox(
    img: np.ndarray,
    bbox: BBox,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    text_scale: float = 1.0,
    thickness: int = 5,
):
    cv2.rectangle(
        img,
        pt1=(bbox.x1, bbox.y1),
        pt2=(bbox.x2, bbox.y2),
        color=color,
        thickness=thickness,
    )
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)

    cv2.rectangle(img, (bbox.x1, bbox.y1 - h), (bbox.x1 + w, bbox.y1), color, -1)
    cv2.putText(
        img,
        label,
        (bbox.x1, bbox.y1 - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        1,
    )
