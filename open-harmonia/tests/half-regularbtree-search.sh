#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
distribution=(uniform normal gamma zipf)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="half-regularbpt"
cd ..
cmake -DCMAKE_BUILD_TYPE=Release -DORDER=8 -DNARROW=2 . 2>&1 > /dev/null 
make 2>&1 > /dev/null 
cd evaluation/
for distr in ${distribution[*]}
do 
    echo "+++++++++++++$distr++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo -e "\n"
        echo "Tree size $ts"
        echo "=======================================================">>${datename}-${distr}-result.$tree_name
        echo "Tree size $ts">>${datename}-${distr}-result.$tree_name

        cpu_time=0
        N=4
        for((i=0;i<$N;i++))
        do 
            j=0
            #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 11 > tmp
            ../regular_bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 8 > tmp
            cat tmp >>${datename}-${distr}-result.$tree_name
 
            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ] 
                then 
                    echo "CPU + $a"
                    cpu_time=$(echo "scale=6; $cpu_time + $a" | bc)
                fi 
                j=$[ $j + 1 ]
            done 
        done 
        echo $cpu_time 

        cpu_time=$(echo "scale=6;$cpu_time / $N" | bc)
        echo "${distr}-${ts}=========Result"
        echo "0$cpu_time" 
     
    done
done 
