#!/bin/bash
tree_size=(23 24 25 26)
distribution=(uniform normal gamma zipf)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="full-multi-noprefix-nosort-8thread"
#tree_name="hb+"
#rm *_result
for distr in ${distribution[*]}
do 
    echo "+++++++++++++$distr++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo -e "\n"
        echo "Tree size $ts"
        echo "=======================================================">>${datename}-${distr}-result.$tree_name
        echo "Tree size $ts">>${datename}-${distr}-result.$tree_name

        GPU1_time=0
        GPU2_time=0
        total_time=0
        N=4
        for((i=0;i<$N;i++))
        do 
            j=0
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 30 > tmp
            cat tmp >>${datename}-${distr}-result.$tree_name

            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2.txt  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ]
                then 
                    echo "GPU1 + $a"
                    GPU1_time=$(echo "scale=6; $GPU1_time + $a" | bc)
                fi 
                if [ $j -eq 1 ]
                then 
                    echo "GPU2 + $a"
                    GPU2_time=$(echo "scale=6; $GPU2_time + $a" | bc)
                fi 
                if [ $j -eq 2 ]
                then 
                    echo "total_time + $a"
                    total_time=$(echo "scale=6; $total_time + $a" | bc)
                fi 
 
                j=$[ $j + 1 ]
            done 
        done 
        echo $GPU1_time 
        echo $GPU2_time
        echo $total_time
        
    

        GPU1_time=$(echo "scale=6;$GPU1_time / $N" | bc)
        GPU2_time=$(echo "scale=6;$GPU2_time / $N" | bc)
        total_time=$(echo "scale=6;$total_time / $N" | bc)
        echo "${distr}-${ts}=========Result"
        echo "0$GPU1_time, 0$GPU2_time, 0$total_time"
        #echo $GPU1_time
        #echo $GPU2_time
        #echo $total_time
        
    done
done 
