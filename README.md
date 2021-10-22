# convert-test-torch2trt


## Startup Docker Containers
```
bash make_env.sh
docker-compose up -d --build 
```

## output model list 
```bash
cd src
bash export_model_list_timm.sh
```

## torch2trt
```bash
python torch2onnx_timm.py --model-name {model_name}
python onnx2trt_timm.py --onnx-path {your/model/path}
```

## inference test
```
python inference_test.py --engine-path {your/engine/path}
```
