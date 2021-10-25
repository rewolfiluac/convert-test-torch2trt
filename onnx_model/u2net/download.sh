# salient
FILE_ID=1V0_AzVxia_UHz8hz93iosU7UqSVg7peT
FILE_NAME=u2net_salient_1_3_320_320.onnx

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

# portrait
FILE_ID=1GuLPvhn4yb1wqMRSC4KSxVYttPpMFPS6
FILE_NAME=u2net_portrait_1_3_512_512.onnx

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
