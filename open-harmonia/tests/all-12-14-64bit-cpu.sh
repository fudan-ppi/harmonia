#!/bin/bash 
dir="1214-64bit-cpu-test"
mkdir $dir 

./link-to-64.sh 


date 
echo search-hb-prefetch 
./cpu-hb-prefetch.sh > ./$dir/result.cpu-hb-prefetch 

date 
echo search-hb-prefetch-sort 
./cpu-hb-prefetch-sort.sh > ./$dir/result.cpu-hb-prefetch-sort 

date 
echo search-hb-prefetch-sort-NTG
./cpu-hb-prefetch-sort-NTG.sh > ./$dir/result.cpu-hb-prefetch-sort-NTG

date 
echo search-harmonia-prefetch
./cpu-harmonia-prefetch.sh > ./$dir/result.cpu-harmonia-prefetch 

date 
echo search-harmonia-prefetch-sort 
./cpu-harmonia-prefetch-sort.sh > ./$dir/result.cpu-harmonia-prefetch-sort 

date
echo search-harmonia-prefetch-sort-NTG
./cpu-harmonia-prefetch-sort-NTG.sh > ./$dir/result.cpu-harmonia-prefetch-sort-NTG
