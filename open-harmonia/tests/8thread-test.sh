#!/bin/bash
tree_size=(23 24 25 26)
distribution=(uniform)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="8thread"
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

        GPU_time=0
        total_time=0
        N=4
        for((i=0;i<$N;i++))
        do 
            j=0
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 5 > tmp
            cat tmp >>${datename}-${distr}-result.$tree_name

            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2.txt  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ]
                then 
                    echo "GPU + $a"
                    GPU_time=$(echo "scale=6; $GPU_time + $a" | bc)
                fi 
                if [ $j -eq 1 ]
                then 
                    echo "total_time + $a"
                    total_time=$(echo "scale=6; $total_time + $a" | bc)
                fi 

                j=$[ $j + 1 ]
            done 
        done 
        echo $GPU_time 
        echo $total_time
        echo $cpu_compute_time
        echo $cpu_total_time   

        GPU_time=$(echo "scale=6;$GPU_time / $N" | bc)
        total_time=$(echo "scale=6;$total_time / $N" | bc)
        echo "${distr}-${ts}=========Result"
        echo $GPU_time 
        echo $total_time
        
    done
done 
