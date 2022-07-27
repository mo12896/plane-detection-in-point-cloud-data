#!/bin/bash

# pull image from docker hub
docker pull mo12896/ransac
# run the image using bind mounts to the data directory
docker run -v /home/moritz/PycharmProjects/plane-detection-in-point-cloud-data/data:/home/data ransac
