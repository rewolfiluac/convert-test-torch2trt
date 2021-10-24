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


def normPRED(d: np.ndarray) -> np.ndarray:
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def postprocess(
    res: List[np.ndarray],
    input_shape: Tuple[int, int],
    resize_shape: Tuple[int, int],
    is_portrait: bool = False,
) -> np.ndarray:
    pred = res[0].reshape((1, *input_shape))
    if is_portrait:
        pred = 1.0 - pred
    pred = normPRED(pred) * 255
    pred = pred.transpose((1, 2, 0)).astype(np.uint8)
    out_img = cv2.resize(pred, resize_shape)
    return out_img


@hydra.main(config_path="../configs", config_name="u2net")
def main(cfg: DictConfig) -> None:
    log.load_config()

    fix_seed(cfg.general.seed)
    is_portrait = True if cfg.u2net.mode == "portrait" else False
    engine_path = Path(cfg.u2net.engine_path)
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
        cfg.u2net.input_shape,
        devide_max=True,
    )
    # 1st inference
    start = time.time()
    load_data(input_data, inputs[0].host)
    _ = do_inference_v2(
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
    out_img = postprocess(
        res,
        input_shape=cfg.u2net.input_shape,
        resize_shape=(img.shape[1], img.shape[0]),
        is_portrait=is_portrait,
    )
    logging.info(f"2nd inference time: {time.time() - start} [sec]")

    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)


if __name__ == "__main__":
    main()
