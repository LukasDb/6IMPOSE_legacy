#!/bin/bash

# MAKE VERSION
# set GPU cudnn libso to 1 in Makefile
sed -i 's/GPU=0/GPU=1/g' darknet/Makefile
sed -i 's/CUDNN=0/CUDNN=1/g' darknet/Makefile
# sed -i 's/OPENCV=0/OPENCV=1/g' darknet/Makefile
# sed -i 's/OPENMP=0/OPENMP=1/g' darknet/Makefile
sed -i 's/LIBSO=0/LIBSO=1/g' darknet/Makefile


# build darknet
cd darknet
make


# CMAKE
# cd darknet
# mkdir build_release
# cd build_release
# cmake ..
# cmake --build . --target install --parallel 8