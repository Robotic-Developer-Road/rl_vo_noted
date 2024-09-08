#!/bin/sh
xhost local:root

XAUTH=/tmp/.docker.xauth
path=/home/$USER
echo "current path: $path"

docker run --privileged --rm -it \
    --volume /etc/groups:/etc/groups:ro \
    --volume /etc/passwd:/etc/passwd:ro \
    --volume $path/rl_vo/:$path/vo_rl/:rw \
    --volume $path/datasets/TartanAir/:$path/datasets/TartanAir/:ro \
    --volume $path/datasets/EuRoC/:$path/datasets/EuRoC/:ro \
    --volume $path/datasets/TUM-RGBD/:$path/datasets/TUM-RGBD/:ro \
    --volume $path/rl_vo/logs/log_voRL/:$path/vo_rl/logs/log_voRL/:rw \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --ipc=host \
    --privileged \
    --user $(id -u):$(id -g $USER) \
    --gpus=all \
    vo_rl
    bash