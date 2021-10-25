#!/bin/bash
function download_from_gdrive(){
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$1" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$1" -o $2
}

# salient
FILE_ID=1V0_AzVxia_UHz8hz93iosU7UqSVg7peT
FILE_NAME=u2net_salient_1_3_320_320.onnx
download_from_gdrive $FILE_ID $FILE_NAME

# portrait
FILE_ID=1GuLPvhn4yb1wqMRSC4KSxVYttPpMFPS6
FILE_NAME=u2net_portrait_1_3_512_512.onnx
download_from_gdrive $FILE_ID $FILE_NAME

# human seg
FILE_ID=1iUnov0RlvgrQJ4H8ip5oJH1YZYySsbhY
FILE_NAME=u2net_human_seg_1_3_320_320.onnx
download_from_gdrive $FILE_ID $FILE_NAME
