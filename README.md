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

## torch2trt
```bash
# torch2onnx
python torch2onnx_timm.py --model-name {model_name}
# onnx2trt
python onnx2trt.py --onnx-path {your/onnx/path}
```

## inference engine
```bash
# timm
python inference_timm.py --engine-path {your/engine/path} --image-path {your/image/path}
# u^2-net
python inference_u2net.py --engine-path {your/engine/path} --image-path {your/image/path}
```
