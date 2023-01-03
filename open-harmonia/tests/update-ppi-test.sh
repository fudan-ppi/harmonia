#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
update_batchs=(16K 32K 64K 128K 256K 512K)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="ppi"
for ub in ${update_batchs[*]}
do 
    echo "+++++++++++++$ub++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo -e "\n"
        echo "Tree size $ts"
        echo "=======================================================">>${datename}-update-${ub}-result.$tree_name
        echo "Tree size $ts">>${datename}-update-${ub}-result.$tree_name

        update_time=0
        N=4
        for((i=0;i<$N;i++))
        do 
            j=0
            #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 11 > tmp
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_uniform.txt 13 ../data_set/update_${ub} > tmp
            cat tmp >>${datename}-update-${ub}-result.$tree_name
 
            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            #for a in $(sed -n '/time/p' 2  | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ] 
                then 
                    echo "update + $a"
                    update_time=$(echo "scale=6; $update_time + $a" | bc)
                fi 
                j=$[ $j + 1 ]
            done 
        done 

        echo $update_time
        update_time=$(echo "scale=6;$update_time / $N" | bc)
        echo "${ts}-${ub}=========Result"
        echo "0$update_time"
     
    done
done 