#!/bin/bash
# Inference timm models
while read line
do
    # fp32
    python torch2onnx_timm.py --model-name $line
    python onnx2trt_timm.py --onnx-path "../onnx_model/${line}.onnx"
    python inference_timm.py --engine-path ../trt_engine/"${line}".engine --image-path ../images/bird.jpg
    # fp16
    python onnx2trt_timm.py --onnx-path "../onnx_model/${line}.onnx" --fp16
    python inference_timm.py --engine-path ../trt_engine/"${line}_fp16".engine --image-path ../images/bird.jpg
done < ../inference_model_list
