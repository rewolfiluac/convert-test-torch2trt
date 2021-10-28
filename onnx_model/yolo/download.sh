#!/bin/bash
function download_from_gdrive(){
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$1" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$1" -o $2
}

# d53 input_608
FILE_ID=1KyhZaJnwuiv14i0qx8p_P7HUnXWTjODy
FILE_NAME=yolov3_d53_mstrain-608_273e_coco.onnx
download_from_gdrive $FILE_ID $FILE_NAME

