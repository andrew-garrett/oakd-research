#! /bin/bash

echo "Cleaning background images"
sudo docker system prune -f

echo "Building Image"
cd oak_capture && sudo docker build -t oak_capture . && cd ..

echo "Starting Container"
xhost local:root
sudo docker-compose up -d
# sudo docker exec -it oakd-research_oak_capture_1 /bin/bash
sudo docker logs oakd-research_oak_capture_1 -f
sudo docker-compose down
