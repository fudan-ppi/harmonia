#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
#distribution=(uniform normal gamma zipf)
distribution=(uniform)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="cpu-range-hb"
for distr in ${distribution[*]}
do 
    echo "+++++++++++++$distr++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo -e "\n"
        echo "Tree size $ts"
        echo "=======================================================">>${datename}-${distr}-result.$tree_name
        echo "Tree size $ts">>${datename}-${distr}-result.$tree_name

        total_time=0
        sort_time=0
        search_time=0
        traverse_time=0
        N=4
        N2=8
        for((i=0;i<$N;i++))
        do 
            j=0
            #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 11 > tmp
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/range_${distr}.txt 62 > tmp
            cat tmp >>${datename}-${distr}-result.$tree_name
 
            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ] 
                then 
                    echo "search time  + $a"
                    search_time=$(echo "scale=6; $search_time + $a" | bc)
                fi 
                if [ $j -eq 1 ] 
                then 
                    echo "traverse time  + $a"
                    traverse_time=$(echo "scale=6; $total_time + $a" | bc)
                fi 
                if [ $j -eq 2 ] 
                then 
                    echo "total time  + $a"
                    total_time=$(echo "scale=6; $total_time + $a" | bc)
                fi 
                j=$[ $j + 1 ]
            done 
        done 
        echo $total_time    

        search_time=$(echo "scale=6;$search_time / $N" | bc)
        traverse_time=$(echo "scale=6;$traverse_time / $N" | bc)
        total_time=$(echo "scale=6;$total_time / $N" | bc)
        echo "${distr}-${ts}=========Result"
        echo "0$search_time"
        echo "0$traverse_time"
        echo "0$total_time" 

     
    done
done 
