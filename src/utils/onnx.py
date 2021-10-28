from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def remove_node_below_nms(onnx_graph: onnx.ModelProto, target: str) -> onnx.ModelProto:
    graph = gs.import_onnx(onnx_graph)
    tensors = graph.tensors()
    layers = [
        node
        for node in graph.nodes
        if node.op == "NonMaxSuppression" and node.name == target
    ]
    if len(layers) > 0:
        layer = layers[0]
        boxes = tensors[f"{layer.inputs[0].name}"].to_variable(dtype=np.float32)
        scores = tensors[f"{layer.inputs[1].name}"].to_variable(dtype=np.float32)
        output_nms = tensors[f"{layer.outputs[0].name}"].to_variable(dtype=np.int64)
        # outputs
        graph.outputs = [boxes, scores, output_nms]

    graph.cleanup()
    out_onnx = gs.export_onnx(graph)
    return out_onnx
