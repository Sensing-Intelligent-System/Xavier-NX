#!/usr/bin/env bash

IMG=argsis/sis:xavier-nx 

xhost +
containerid=$(docker ps -aqf "ancestor=${IMG}")&& echo $containerid
docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it sis bash
xhost -
