#!/bin/bash 
dir="910update-8core"
mkdir $dir  

date
echo update-rebuild-ppi-test
./update-rebuild-ppi-test.sh >./$dir/result.update-rebulid-ppi-test-noleaf
date 
echo update-hb+ 
./update-test.sh >./$dir/result.update-hb+
date
echo update-regular
./update-regular-bpt.sh >./$dir/result.update-regularB+
date
