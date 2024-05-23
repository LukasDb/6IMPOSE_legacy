#FROM ubuntu:22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install curl wget git make build-essential cmake libopencv-dev -y

# Install darknet
WORKDIR /app/
RUN git clone https://github.com/AlexeyAB/darknet
COPY build_darknet.sh /app/build_darknet.sh
RUN chmod +x /app/build_darknet.sh
RUN /app/build_darknet.sh

RUN wget micro.mamba.pm/install.sh
RUN bash install.sh

RUN /root/.local/bin/micromamba create -n tf python=3.6.5 -y

# other dependencies
RUN /root/micromamba/envs/tf/bin/pip install tensorflow==2.6.2
RUN /root/micromamba/envs/tf/bin/pip install --no-cache-dir flask pillow matplotlib
RUN /root/micromamba/envs/tf/bin/pip install --no-cache-dir --only-binary opencv-python opencv-python
# down grade protobuf to 3.20
# RUN /root/micromamba/envs/tf/bin/pip install protobuf==3.20.*

# compile tf ops
COPY tf_ops /app/tf_ops
RUN chmod +x /app/tf_ops/compile_ops.sh
WORKDIR /app/tf_ops
RUN ./compile_ops.sh
WORKDIR /app/

# SETUP YOLO
COPY yolo.py /app/yolo.py
COPY weights /app/weights

# SETUP PVN3D
COPY pvn3d.py /app/pvn3d.py
COPY keypoints /app/keypoints

# Copy main.py and entrypoint.sh
COPY main.py /app/main.py
COPY entrypoint.sh /app/entrypoint.sh

# Set the permissions
RUN chmod +x /app/entrypoint.sh

# Set the working directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

