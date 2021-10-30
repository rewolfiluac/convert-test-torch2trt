from pathlib import Path
import time
from typing import List, Any, Tuple
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2

from img_proc.np import preprocess
from utils.onnx import nms_postprocess
from utils import log
from img_proc.bbox import BBox, draw_bboxs, correct_bbox
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
    engine_path = Path(cfg.yolo.engine_path)
    img_path = Path(cfg.general.image_path)
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if not engine_path.is_file():
        raise Exception(f"File Not Found. {str(engine_path)}")

    # prepare tensorrt
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # prepare intput data
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

    # prepare output
    out_img = img.copy()

    bboxes = correct_bbox(
        out_img, out_boxes[0], cfg.yolo.input_shape[0], cfg.yolo.input_shape[1]
    )
    draw_bboxs(
        out_img,
        COCO_CLS_LABELS,
        bboxes,
        out_classes.tolist(),
    )
    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)


if __name__ == "__main__":
    main()
