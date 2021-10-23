import argparse
from pathlib import Path
import time
from typing import List
import logging

import numpy as np
import cv2

from utils import log
from utils.util import fix_seed
from utils.trt import (
    load_engine,
    allocate_buffers,
    do_inference_v2,
    load_data,
)
from consts.imagenet_labels import IMAGENET_LABELS

IMAGE_DIR = "../images"


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def get_dummy_input(
    batch_size: int,
    color_size: int,
    hetigh_size: int,
    width_size: int,
) -> np.ndarray:
    input_shape_size = (
        batch_size,
        color_size,
        hetigh_size,
        width_size,
    )
    input_data = np.random.randint(0, 255, input_shape_size)
    return input_data


def preprocess(input_data: np.ndarray) -> np.ndarray:
    input_data = cv2.resize(input_data, (224, 224))
    # imagenet color RGB, not BGR.
    input_data = input_data[:, :, ::-1]
    # (h, w, c) to (c, h, w)
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, 0)
    input_data = input_data.astype(np.float32)
    input_data = input_data / 255
    input_data[:, 0, :, :] = (input_data[:, 0, :, :] - 0.485) / 0.229
    input_data[:, 1, :, :] = (input_data[:, 1, :, :] - 0.456) / 0.224
    input_data[:, 2, :, :] = (input_data[:, 2, :, :] - 0.406) / 0.225
    return input_data.ravel()


if __name__ == "__main__":
    args = get_argparser()
    log.load_config()

    fix_seed(args.seed)
    engine_path = Path(args.engine_path)
    img_path = Path(args.image_path)
    if not engine_path.is_file():
        raise Exception(f"File Not Found. {str(engine_path)}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    img = cv2.imread(str(img_path))
    input_data = preprocess(img)

    start = time.time()
    load_data(input_data, inputs[0].host)

    res_prob = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    res_cls = res_prob[0].reshape((1, 1000)).argmax()
    logging.info(f"inference time: {time.time() - start} [sec]")

    logging.info(f"pred: {IMAGENET_LABELS[res_cls]}")
