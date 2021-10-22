import argparse
from pathlib import Path
from typing import Tuple

import timm
import torch

OUT_DIR_PATH = "../onnx"


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--input-batch", type=int, default=1)
    parser.add_argument("--input-color", type=int, default=3)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--model-list", help="show model list.", action="store_true")
    args = parser.parse_args()
    return args


def show_model_list() -> None:
    model_names = timm.list_models(pretrained=True)
    for model_name in model_names:
        print(model_name)


def torch2onnx(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int],
    output_path: Path,
) -> None:
    dummy_input = torch.randn(*input_size, requires_grad=True)
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        str(output_path),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    args = get_argparser()
    if args.model_list:
        show_model_list()
    else:
        out_p = Path(OUT_DIR_PATH) / f"{args.model_name}.onnx"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        input_shape_size = (
            args.input_batch,
            args.input_color,
            args.input_height,
            args.input_width,
        )
        model = timm.create_model(args.model_name, pretrained=True)
        torch2onnx(
            model=model,
            input_size=input_shape_size,
            output_path=out_p,
        )
