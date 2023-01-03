#!/bin/bash
cwd=$(pwd)
cd ./script/start-docker
sh build.sh
sh start.sh
