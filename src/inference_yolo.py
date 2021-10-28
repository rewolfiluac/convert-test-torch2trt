from pathlib import Path
import time
from typing import List, Any, Tuple
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2

from img_proc.np import preprocess
from utils import log
from utils.util import fix_seed
from utils.trt import (
    load_engine,
    allocate_buffers,
    do_inference_v2,
    load_data,
)
from consts.imagenet_labels import IMAGENET_LABELS

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
    boxes = ret[0].reshape((1, -1, 4))
    scores = ret[1].reshape((1, 80, -1))
    selected_indices = ret[2].reshape((1, -1, 3))

    boxes_selected = np.take(boxes, selected_indices[0, :, 2], axis=1)
    scores_selected = scores[0, selected_indices[0, :, 1], selected_indices[0, :, 2]]
    idxs = list(zip(*np.where(scores_selected > cfg.yolo.score_thr)))
    selected_box_indices = selected_indices[:, idxs, 2]
    show_boxes = np.take(
        boxes, selected_box_indices.reshape(selected_box_indices.shape[:2]), axis=1
    )
    logging.info(f"2nd postprocess time: {time.time() - start} [sec]")

    # Show BBox
    org_h, org_w = img.shape[:2]
    line_size = org_h if org_h < org_w else org_w
    line_size = line_size // 100
    h_ratio = org_h / cfg.yolo.input_shape[0]
    w_ratio = org_w / cfg.yolo.input_shape[0]

    out_img = img.copy()
    for box in show_boxes[0, 0]:
        cv2.rectangle(
            out_img,
            pt1=(int(box[0] * w_ratio), int(box[1] * h_ratio)),
            pt2=(int(box[2] * w_ratio), int(box[3] * h_ratio)),
            color=(0, 255, 0),
            thickness=line_size,
            # lineType=cv2.LINE_4,
            shift=0,
        )

    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)


if __name__ == "__main__":
    main()
