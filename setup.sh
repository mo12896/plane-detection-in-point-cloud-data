#!/bin/bash

# pull image from docker hub
docker pull mo12896/ransac
# run the image using bind mounts to the data directory
local_data_dir="/home/moritz/PycharmProjects/plane-detection-in-point-cloud-data/data"
docker run -v $local_data_dir:/home/data ransac
