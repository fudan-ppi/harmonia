#!/bin/bash 
dir="titan3"
mkdir $dir 


date 
echo full-prefix-nosort-1thread
./full-multi-prefix-nosort-1thread.sh > ./${dir}/result.full-multi-prefix-nosort-1thread
date 
echo full-prefix-nosort-8thread
./full-multi-prefix-nosort-8thread.sh > ./$dir/result.full-multi-prefix-nosort-8thread

date 
echo full-prefix-sort-1thread
./full-multi-prefix-sort-1thread.sh > ./$dir/result.full-multi-prefix-sort-1thread
date 
echo full-prefix-sort-8thread
./full-multi-prefix-sort-8thread.sh > ./$dir/result.full-multi-prefix-sort-8thread


date
