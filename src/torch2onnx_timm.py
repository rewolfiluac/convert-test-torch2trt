from pathlib import Path
from typing import Tuple
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import timm
import torch
import onnx

from utils import log


OUT_DIR_PATH = "../onnx_model"


def get_dummy_input(
    batch_size: int,
    color_size: int,
    hetigh_size: int,
    width_size: int,
) -> torch.Tensor:
    input_shape_size = (
        batch_size,
        color_size,
        hetigh_size,
        width_size,
    )
    return torch.randn(*input_shape_size, requires_grad=True)


def show_model_list() -> None:
    model_names = timm.list_models(pretrained=True)
    for model_name in model_names:
        print(model_name)


@log.start_end_log
def torch2onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> None:
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        str(output_path),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


@log.start_end_log
def check_onnx(onnx_path: Path) -> None:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


@hydra.main(
    config_path="../configs",
    config_name="torch2onnx",
)
def main(cfg: DictConfig) -> None:
    if cfg.general.show_model_list:
        show_model_list()
        exit()

    logging.info(f"Start Convert - Model Name : {cfg.timm.model_name}")

    out_p = Path(OUT_DIR_PATH) / f"{cfg.timm.model_name}.onnx"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = get_dummy_input(1, 3, *cfg.timm.input_shape)
    model = timm.create_model(cfg.timm.model_name, pretrained=True)
    torch2onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=out_p,
    )
    check_onnx(out_p)
    logging.info(f"Complete Convert")


if __name__ == "__main__":
    main()
