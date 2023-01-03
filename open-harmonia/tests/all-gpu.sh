#!/bin/bash 
dir="0713"
mkdir $dir 
mkdir $dir/update 
mkdir $dir/narrow

echo multi-regular-narrow
./multi-regular-narrow.sh ${dir} 

data
echo hbtree
./hbtree-search.sh > ./$dir/result.hb+
date 
echo prefix-nosort-1thread
./multi-prefix-nosort-1thread.sh > ./${dir}/result.multi-prefix-nosort-1thread
date 
echo prefix-nosort-2thread
./multi-prefix-nosort-2thread.sh > ./$dir/result.multi-prefix-nosort-2thread
date 
echo prefix-nosort-4thread
./multi-prefix-nosort-4thread.sh > ./$dir/result.multi-prefix-nosort-4thread
date 
echo prefix-nosort-8thread
./multi-prefix-nosort-8thread.sh > ./$dir/result.multi-prefix-nosort-8thread

date 
echo prefix-sort-1thread
./multi-prefix-sort-1thread.sh > ./$dir/result.multi-prefix-sort-1thread
date 
echo prefix-sort-2thread
./multi-prefix-sort-2thread.sh > ./$dir/result.multi-prefix-sort-2thread
date 
echo prefix-sort-4thread
./multi-prefix-sort-4thread.sh > ./$dir/result.multi-prefix-sort-4thread
date 
echo prefix-sort-8thread
./multi-prefix-sort-8thread.sh > ./$dir/result.multi-prefix-sort-8thread

date
