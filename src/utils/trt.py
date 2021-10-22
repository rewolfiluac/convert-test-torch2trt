from pathlib import Path
from typing import Any

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val: int) -> int:
    return val * 1 << 30


class HostDeviceMem(object):
    def __init__(self, host_mem: Any, device_mem: Any) -> None:
        self.host = host_mem
        self.device = device_mem

    def __str__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self) -> str:
        return self.__str__()


def build_engine(onnx_path: Path, is_fp16: bool = False) -> Any:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = GiB(1)
    if is_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_geterror(error))
            return None
    return builder.build_engine(network, config)
