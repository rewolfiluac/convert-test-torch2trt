from pathlib import Path
import time
from typing import List, Any, Tuple
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2

from img_proc.padding import calc_pad_size, pad
from img_proc.np import preprocess
from utils.onnx import nms_postprocess
from utils import log
from utils.util import fix_seed, draw_bbox
from utils.trt import (
    load_engine,
    allocate_buffers,
    do_inference_v2,
    load_data,
)
from consts.coco_labels import COCO_CLS_LABELS

OUTPUT_DIR = "../images_out"


@hydra.main(config_path="../configs", config_name="yolo")
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.general.seed)
    engine_path = Path(cfg.yolo.engine_path)
    img_path = Path(cfg.general.image_path)
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if not engine_path.is_file():
        raise Exception(f"File Not Found. {str(engine_path)}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    img = cv2.imread(str(img_path))
    input_data = preprocess(
        img,
        cfg.yolo.input_shape,
        padding=cfg.yolo.input_padding,
        mean=cfg.yolo.normalize.mean,
        std=cfg.yolo.normalize.std,
    )

    # pre inference
    load_data(input_data, inputs[0].host)
    ret = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    logging.info(f"1st inference done.")

    # preprocess
    start = time.time()
    input_data = preprocess(
        img,
        cfg.yolo.input_shape,
        padding=cfg.yolo.input_padding,
        mean=cfg.yolo.normalize.mean,
        std=cfg.yolo.normalize.std,
    )
    logging.info(f"2nd preprocess time: {time.time() - start} [sec]")

    # inferendce
    start = time.time()
    load_data(input_data, inputs[0].host)
    ret = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    logging.info(f"2nd inference time: {time.time() - start} [sec]")

    # postprocess
    start = time.time()
    out_boxes, out_classes = nms_postprocess(
        ret[0],
        ret[1],
        ret[2],
        score_thr=cfg.yolo.score_thr,
    )
    logging.info(f"2nd postprocess time: {time.time() - start} [sec]")

    # Draw BBox
    out_img = img.copy()
    org_h, org_w = out_img.shape[:2]
    # Padding分の計算を加える
    line_size = org_h if org_h < org_w else org_w
    line_size = line_size // 100

    tblr = calc_pad_size(*out_img.shape[:2])
    pad_h, pad_w = pad(out_img, tblr).shape[:2]

    # normalize bbox for padding and resize
    h_ratio = pad_h / (cfg.yolo.input_shape[0])
    w_ratio = pad_w / (cfg.yolo.input_shape[1])
    h_pad_diff = tblr[0]
    w_pad_diff = tblr[2]

    for box, cls_idx in zip(out_boxes[0], out_classes):
        _box = [
            int(box[0] * w_ratio - w_pad_diff),
            int(box[1] * h_ratio - h_pad_diff),
            int(box[2] * w_ratio - w_pad_diff),
            int(box[3] * h_ratio - h_pad_diff),
        ]
        draw_bbox(
            out_img,
            _box,
            COCO_CLS_LABELS[cls_idx + 1],
            text_scale=0.5,
            thickness=3,
        )

    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)


if __name__ == "__main__":
    main()
