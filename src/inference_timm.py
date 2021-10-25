import argparse
from pathlib import Path
import time
from typing import List
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2

from img_proc.np import preprocess
from utils.util import fix_seed
from utils.trt import (
    load_engine,
    allocate_buffers,
    do_inference_v2,
    load_data,
)
from consts.imagenet_labels import IMAGENET_LABELS


@hydra.main(config_path="../configs", config_name="timm")
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.general.seed)
    engine_path = Path(cfg.timm.engine_path)
    img_path = Path(cfg.general.image_path)
    if not engine_path.is_file():
        raise Exception(f"File Not Found. {str(engine_path)}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    img = cv2.imread(str(img_path))
    input_data = preprocess(
        img,
        resize_shape=cfg.timm.input_shape,
    )

    # 1st inference
    start = time.time()
    load_data(input_data, inputs[0].host)

    res_prob = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    res_cls = res_prob[0].reshape((1, cfg.timm.classes_num)).argmax()
    logging.info(f"1st inference time: {time.time() - start} [sec]")

    # 2nd inference
    start = time.time()
    load_data(input_data, inputs[0].host)

    res_prob = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    res_cls = res_prob[0].reshape((1, cfg.timm.classes_num)).argmax()
    logging.info(f"2nd inference time: {time.time() - start} [sec]")

    logging.info(f"pred: {IMAGENET_LABELS[res_cls]}")

    del context, bindings, inputs, outputs, stream


if __name__ == "__main__":
    main()
