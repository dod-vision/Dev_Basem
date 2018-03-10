# Python 
Python 3+

# Dependancies
- opencv 3
- [darkflow](https://github.com/thtrieu/darkflow)

## Running demo		
To run 
``` 	
python main.py --videoCam video.mp4 --record
	
--record : save the output video in the project directory	
```


# Docker

## Acessar pasta do projeto

## Create docker image
```
docker build -t dod .
```
## Get project directory
```
pwd
```
## Create and start container
```
docker run -it -v <project_directory>:/app -p 8899:8899 --name dod_test dod /bin/bash

docker run -it --rm --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <project_directory>:/app -p 8899:8899 --name dod_run dod /bin/bash
```
