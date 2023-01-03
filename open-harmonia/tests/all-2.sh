#!/bin/bash 
dir="titan2"
mkdir $dir 
mkdir $dir/update 
mkdir $dir/regularB+

date
echo update-rebuild-ppi-test
./update-rebuild-ppi-test.sh >./$dir/update/result.update-rebulid-ppi-test
date 
echo multi-regular-base-gpu
./multi-regular-base-gpu.sh ${dir} 
date
echo multi-regular-prefix-nosort
./multi-regular-prefix-nosort.sh ${dir} 
date
echo multi-regular-prefix-sort
./multi-regular-prefix-sort.sh ${dir} 
date
echo half-multi-regular-base-gpu
./half-multi-regular-base-gpu.sh ${dir}
date 
echo half-multi-regularbtree
./half-regularbtree-search.sh > ./${dir}/regular.half_regularbtree-search
date 
echo half_multi_regular_prefix_nosort
./half-multi-regular-prefix-nosort.sh ${dir}
date 
echo half_multi_regular_prefix_sort
./half-multi-regular-prefix-sort.sh ${dir}
date
