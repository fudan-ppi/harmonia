#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
distribution=(uniform normal gamma zipf)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="hb+"
for distr in ${distribution[*]}
do 
    echo "+++++++++++++$distr++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo -e "\n"
        echo "Tree size $ts"
        echo "=======================================================">>${datename}-${distr}-result.$tree_name
        echo "Tree size $ts">>${datename}-${distr}-result.$tree_name

        cpu_up_time=0
        GPU_time=0
        cpu_down_time=0
        total_time=0
        N=4
        N2=8
        for((i=0;i<$N;i++))
        do 
            j=0
            #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 11 > tmp
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 11 > tmp
            cat tmp >>${datename}-${distr}-result.$tree_name
 
            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ] 
                then 
                    echo "CPU up + $a"
                    cpu_up_time=$(echo "scale=6; $cpu_up_time + $a" | bc)
                fi 
                if [ $j -eq 1 ] || [ $j -eq 2 ]
                then 
                    echo "GPU + $a"
                    GPU_time=$(echo "scale=6; $GPU_time + $a" | bc)
                fi 
                if [ $j -eq 3 ]
                then 
                    echo "cpu_down_time + $a"
                    cpu_down_time=$(echo "scale=6; $cpu_down_time + $a" | bc)
                fi 
                if [ $j -eq 4 ]
                then 
                    echo "total_time + $a"
                    total_time=$(echo "scale=6; $total_time + $a" | bc)
                fi 

                j=$[ $j + 1 ]
            done 
        done 
        echo $cpu_up_time 
        echo $GPU_time 
        echo $cpu_down_time 
        echo $total_time    

        cpu_up_time=$(echo "scale=6;$cpu_up_time / $N" | bc)
        GPU_time=$(echo "scale=6;$GPU_time / $N2" | bc)
        cpu_down_time=$(echo "scale=6; $cpu_down_time / $N" | bc)
        total_time=$(echo "scale=6;$total_time / $N" | bc)
        echo "${distr}-${ts}=========Result"
        echo "0$cpu_up_time,0$GPU_time,0$cpu_down_time,0$total_time" 
        #echo $GPU_time 
        #echo $cpu_down_time 
        #echo $total_time    
     
    done
done 
