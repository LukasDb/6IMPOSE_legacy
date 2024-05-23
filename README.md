# 6IMPOSE legacy
Easy to use docker container to run the PVN3D model trained in the [6IMPOSE](https://www.frontiersin.org/articles/10.3389/frobt.2023.1176492/full) project.
If you use this work please cite
```
@ARTICLE{cao2023,
AUTHOR={Cao, Hongpeng and Dirnberger, Lukas and Bernardini, Daniele and Piazza, Cristina and Caccamo, Marco},   
TITLE={6IMPOSE: bridging the reality gap in 6D pose estimation for robotic grasping},      
JOURNAL={Frontiers in Robotics and AI},      
VOLUME={10},           
YEAR={2023},      
URL={https://www.frontiersin.org/articles/10.3389/frobt.2023.1176492},       
DOI={10.3389/frobt.2023.1176492},      
ISSN={2296-9144}
}
```

## Requirements
- Docker
- Nvidia GPU with CUDA support
- Nvidia Container Toolkit


## Usage
Build with:
```
sudo docker build -t simpose .
```
Run with:
```
sudo docker run --runtime=nvidia --gpus=all -p 5000:5000 --rm --name=simpose -it simpose
```
Instead you can use the convience script (make ./run.sh executable with `chmod +x run.sh`):
```
./run.sh
```
