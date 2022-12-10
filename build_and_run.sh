echo Building Latest Image
docker build -t custom_oak_image .
docker run --rm \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    --device-cgroup-rule='c 189:* rmw' \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    custom_oak_image:latest \
    python3 main.py
