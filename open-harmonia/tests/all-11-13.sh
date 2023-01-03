#!/bin/bash 
dir="1113"
mkdir $dir 
mkdir $dir/update 
mkdir $dir/narrow

#date 
#echo regularB+
#./regularbtree-search.sh > ./$dir/result.regularB+
# transfer
echo GPU-notrans----------------------------------------------------------------------------------------------
date 
echo prefix-sort-1thread
./full-multi-prefix-sort-1thread-notransfer.sh > ./$dir/result.full-multi-prefix-sort-1thread-notrans-uniform
date 
echo prefix-sort-8thread
./full-multi-prefix-sort-8thread-notransfer.sh > ./$dir/result.full-multi-prefix-sort-8thread-notrans-uniform 
date 
echo prefix-nosort-1thread
./full-multi-prefix-nosort-1thread-notransfer.sh > ./${dir}/result.full-multi-prefix-nosort-1thread-notrans-uniform 
date 
echo prefix-nosort-8threadfads
./full-multi-prefix-nosort-8thread-notransfer.sh > ./$dir/result.full-multi-prefix-nosort-8thread-notrans-uniform
date 
echo hbtree-simple-search-notrans 
./hbtree-simple-search-notrans.sh > ./$dir/result.hb+-notrans-uniform



#date 
#echo prefix-sort-1thread
#./multi-prefix-sort-1thread.sh > ./$dir/result.multi-prefix-sort-1thread
#date 
#echo prefix-sort-8thread
#./multi-prefix-sort-8thread.sh > ./$dir/result.multi-prefix-sort-8thread
#date 
#echo prefix-nosort-1thread
#./multi-prefix-nosort-1thread.sh > ./${dir}/result.multi-prefix-nosort-1thread
#date 
#echo prefix-nosort-8thread
#./multi-prefix-nosort-8thread.sh > ./$dir/result.multi-prefix-nosort-8thread
#date
#echo hbtree
#./hbtree-search.sh > ./$dir/result.hb+

echo CPU-----------------------------------------------------------------------------------------------------
date
echo cpu-hbtree
./cpu-hb-search.sh > ./$dir/result.cpu-hb+
date
echo cpu-harmonia
./cpu-harmonia-search.sh > ./$dir/result.cpu-harmonia+
date
echo regularbtree-search 
./regularbtree-search.sh > ./$dir/result.cpu-regularbtree-search


#echo multi-regular-narrow
#./multi-regular-narrow.sh ${dir} 



