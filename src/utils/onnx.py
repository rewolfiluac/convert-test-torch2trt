from typing import Tuple

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


def nms_postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    selected_indices: np.ndarray,
    score_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # boxes shape is [num_batches, spatial_dimension, 4].
    boxes = boxes.reshape((1, -1, 4))
    # scores shape is [num_batches, num_classes, spatial_dimension].
    scores = scores.reshape((1, 80, -1))
    # selected_indices format is [batch_index, class_index, box_index].
    # shape is [select_num, 3]
    selected_indices = selected_indices.reshape((1, -1, 3))

    batch_size = boxes.shape[0]

    boxes_selected = np.take(boxes, selected_indices[:, :, 2], axis=1)
    boxes_selected = boxes_selected.reshape(
        (boxes.shape[0], *boxes_selected.shape[-2:])
    )
    scores_selected = scores[
        :, selected_indices[:, :, 1], selected_indices[:, :, 2]
    ].reshape(batch_size, -1)
    idxs = np.where(scores_selected > score_thr)
    selected_box_indices = selected_indices[idxs[0], idxs[1], 2]
    selected_cls_indices = selected_indices[idxs[0], idxs[1], 1]
    out_boxes = boxes[:, selected_box_indices]
    out_classes = selected_cls_indices
    return out_boxes, out_classes
