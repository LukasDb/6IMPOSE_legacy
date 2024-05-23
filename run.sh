#!/bin/bash

docker build -t simpose .

if [ $? -eq 0 ]; then
    echo "Build successful. Continuing..."
    docker run --runtime=nvidia --gpus=all -p 5000:5000 --rm --name=simpose -it simpose
else
    echo "Build failed. Exiting..."
fi
