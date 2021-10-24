# convert-test-torch2trt

## Startup docker container
```
bash make_env.sh
docker-compose up -d --build 
```

## show timm model list 
```bash
cd src
bash export_model_list_timm.sh
```

## torch2trt from timm model
```bash
python torch2onnx_timm.py --model-name {model_name}
```

## download u2net onnx model
```bash

```

## onnx2trt
```bash
python onnx2trt.py --onnx-path {your/onnx/path} {option: --fp16}
```

## inference demo imagenet
```bash
# timm model
python inference_timm.py
# timm model
python inference_timm.py general.image_path={your/image/path}
```
## inference demo u2net
```bash
# u^2-net for portrait
python inference_u2net.py
# (option) u^2-net for salient object detection
python inference_u2net.py u2net=salient
# u^2-net inference your image
python inference_u2net.py general.image_path={your/image/path}
```
