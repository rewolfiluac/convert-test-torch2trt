from pathlib import Path
import time
from typing import List, Any, Tuple
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

OUTPUT_DIR = "../images_out"


def mask_portrait(
    org_img: np.ndarray,
    mask_img: np.ndarray,
) -> np.ndarray:
    inv_mask_img = 255 - mask_img
    mask = mask_img / 255
    res_img = np.empty(org_img.shape, dtype=np.float32)
    res_img[:, :, 0] = org_img[:, :, 0] * mask + inv_mask_img
    res_img[:, :, 1] = org_img[:, :, 1] * mask + inv_mask_img
    res_img[:, :, 2] = org_img[:, :, 2] * mask + inv_mask_img
    return res_img.astype(np.uint8)


def norm_hist(img: np.ndarray) -> np.ndarray:
    # histgram normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


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


@hydra.main(
    config_path="../configs",
    config_name="u2net_portrait",
)
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.general.seed)
    salient_engine_path = Path(cfg.salient.engine_path)
    portrait_engine_path = Path(cfg.portrait.engine_path)
    img_path = Path(cfg.general.image_path)
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    if not salient_engine_path.is_file():
        raise Exception(f"File Not Found. {str(salient_engine_path)}")
    if not portrait_engine_path.is_file():
        raise Exception(f"File Not Found. {str(portrait_engine_path)}")

    # prepare salient engine
    salient_engine = load_engine(salient_engine_path)
    salient_context = salient_engine.create_execution_context()
    (
        salient_inputs,
        salient_outputs,
        salient_bindings,
        salient_stream,
    ) = allocate_buffers(salient_engine)
    # prepare portrait engine
    portrait_engine = load_engine(portrait_engine_path)
    portrait_context = portrait_engine.create_execution_context()
    (
        portrait_inputs,
        portrait_outputs,
        portrait_bindings,
        portrait_stream,
    ) = allocate_buffers(portrait_engine)

    img = cv2.imread(str(img_path))

    # inference salient
    input_data = preprocess(
        img,
        cfg.salient.input_shape,
        devide_max=True,
    )
    load_data(input_data, salient_inputs[0].host)
    res = do_inference_v2(
        context=salient_context,
        bindings=salient_bindings,
        inputs=salient_inputs,
        outputs=salient_outputs,
        stream=salient_stream,
    )
    mask_img = postprocess(
        res,
        input_shape=cfg.salient.input_shape,
        resize_shape=(img.shape[1], img.shape[0]),
    )

    if cfg.general.norm_hist:
        img = norm_hist(img)

    masked_img = mask_portrait(img, mask_img)

    if cfg.general.save_masked_img:
        cv2.imwrite(str(out_dir_path / "masked.jpg"), masked_img)

    # 2nd inference
    input_data = preprocess(
        masked_img,
        cfg.portrait.input_shape,
        devide_max=True,
    )
    load_data(input_data, portrait_inputs[0].host)
    res = do_inference_v2(
        context=portrait_context,
        bindings=portrait_bindings,
        inputs=portrait_inputs,
        outputs=portrait_outputs,
        stream=portrait_stream,
    )
    out_img = postprocess(
        res,
        input_shape=cfg.portrait.input_shape,
        resize_shape=(img.shape[1], img.shape[0]),
        is_portrait=True,
    )

    # save image
    cv2.imwrite(str(out_dir_path / img_path.name), out_img)


if __name__ == "__main__":
    main()
