import argparse
from pathlib import Path
import time
from typing import List, Any, Tuple
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

OUTPUT_DIR = "../images_out"


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
    input_data = cv2.resize(input_data, (320, 320))
    # imagenet color RGB, not BGR.
    input_data = input_data[:, :, ::-1]
    # (h, w, c) to (c, h, w)
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, 0)
    input_data = input_data.astype(np.float32)
    input_data = input_data / np.max(input_data)
    input_data[:, 0, :, :] = (input_data[:, 0, :, :] - 0.485) / 0.229
    input_data[:, 1, :, :] = (input_data[:, 1, :, :] - 0.456) / 0.224
    input_data[:, 2, :, :] = (input_data[:, 2, :, :] - 0.406) / 0.225
    return input_data.ravel()


def normPRED(d: np.ndarray) -> np.ndarray:
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def postprocess(
    res: List[np.ndarray],
    out_width: int,
    out_height: int,
) -> np.ndarray:
    pred = res[0].reshape((1, 320, 320))
    pred = normPRED(pred) * 255
    pred = pred.transpose((1, 2, 0)).astype(np.uint8)
    out_img = cv2.resize(pred, (out_width, out_height))
    return out_img


def predict(
    context: Any,
    bindings: Any,
    inputs: Any,
    outputs: Any,
    stream: Any,
) -> List[np.ndarray]:
    res = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    return res


if __name__ == "__main__":
    args = get_argparser()
    log.load_config()

    fix_seed(args.seed)
    engine_path = Path(args.engine_path)
    img_path = Path(args.image_path)
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if not engine_path.is_file():
        raise Exception(f"File Not Found. {str(engine_path)}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    img = cv2.imread(str(img_path))
    input_data = preprocess(img)
    # 1st inference
    start = time.time()
    load_data(input_data, inputs[0].host)
    _ = predict(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    logging.info(f"1st inference time: {time.time() - start} [sec]")

    # 2nd inference
    start = time.time()
    load_data(input_data, inputs[0].host)
    res = do_inference_v2(
        context=context,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    out_img = postprocess(res, img.shape[1], img.shape[0])
    logging.info(f"2nd inference time: {time.time() - start} [sec]")

    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)
