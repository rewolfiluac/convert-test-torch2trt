import argparse
from pathlib import Path

from utils.trt import build_engine

OUT_DIR_PATH = "../trt_engine"


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=str)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    return args


def get_trt_file_name(onnx_file_name: str, fp16: bool) -> str:
    if fp16:
        return f"{onnx_file_name}_fp16.engine"
    return f"{onnx_file_name}.engine"


if __name__ == "__main__":
    args = get_argparser()

    onnx_path = Path(args.onnx_path)
    trt_path = Path(OUT_DIR_PATH) / get_trt_file_name(
        onnx_path.stem,
        args.fp16,
    )
    trt_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.is_file():
        raise Exception(f"File Not Found. {str(onnx_path)}")

    # build engine
    engine = build_engine(onnx_path=onnx_path, fp16=args.fp16)
    print("Complete build.")
    # write engine file
    with open(str(trt_path), "wb") as f:
        f.write(bytearray(engine.serialize()))
    print("Complete write engine.")
