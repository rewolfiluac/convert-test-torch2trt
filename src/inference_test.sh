#!/bin/bash
while read line
do
    python torch2onnx_timm.py --model-name $line
    python onnx2trt_timm.py --onnx-path "../onnx_model/${line}.onnx"
    python inference_test.py --engine-path ../trt_engine/"${line}".engine --image-path ../images/bird.jpg
done < ../model_list
