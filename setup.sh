#!/bin/bash

# pull image from docker hub
docker pull mo12896/ransac
# define the data dir variable
data="/data"
curr_dir="$PWD"
data_dir=$curr_dir$data
# run the image using bind mounts to the data directory
docker run -v $data_dir:/home/data ransac
