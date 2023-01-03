#!/bin/bash
# This shell is used to generate data files. All generated data will be in ./script/generate-data/dataset
cwd=$(pwd)
echo "generate.sh: please wait, this usually takes about 3 minutes\n"
cd ./script/generate-data
python generate.py
