from pathlib import Path
from typing import Any, List

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
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


def build_engine(onnx_path: Path, fp16: bool = False) -> Any:
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network(EXPLICIT_BATCH) as network:
            with builder.create_builder_config() as config:
                with trt.OnnxParser(network, TRT_LOGGER) as parser:
                    config.max_workspace_size = GiB(1)
                    if fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
                    with open(onnx_path, "rb") as model:
                        if not parser.parse(model.read()):
                            for error in range(parser.num_errors):
                                print(parser.get_error(error))
                            return None
                    return builder.build_engine(network, config)


def load_engine(engine_path: Path) -> Any:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(str(engine_path), "rb") as f:
        engine_byte = f.read()
    return runtime.deserialize_cuda_engine(engine_byte)


def allocate_buffers(engine: Any) -> Any:
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def load_data(data: np.ndarray, pagelocked_buffer: np.ndarray) -> None:
    np.copyto(pagelocked_buffer, data)


def do_inference_v2(
    context: Any,
    bindings: Any,
    inputs: Any,
    outputs: Any,
    stream: Any,
) -> List[Any]:
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
