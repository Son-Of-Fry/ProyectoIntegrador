#!/bin/bash

docker stop qt-yolo 2>/dev/null
docker rm qt-yolo 2>/dev/null

xhost +local:docker

docker run -it --rm \
    --name qt-yolo \
    --gpus all \
    --device /dev/video0:/dev/video0 \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    qt-yolo

xhost -local:docker
    