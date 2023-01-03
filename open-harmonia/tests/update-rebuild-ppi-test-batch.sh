#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
#syn_batches=(32000 64000 128000 256000 512000 1024000 2048000 4096000)
#rebuild_batches=(256000 512000 1024000 2048000 4096000)

syn_batches=(8000 16000 32000 64000)
rebuild_batches=(8000 16000 32000 64000)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="rebuild-ppi-batch"
for syn in ${syn_batches[*]}
do 
    for rebuild in ${rebuild_batches[*]}
    do
        if [ $rebuild -lt $syn ]
        then 
            continue 
        fi
        echo "+++++++++++++syn: $syn  rebuild: $rebuild++++++++++++++"
        for ts in ${tree_size[*]}
        do 
            echo -e "\n"
            echo "Tree size $ts"
            echo "=======================================================">>${datename}-update-${syn}-${rebuild}-result.$tree_name
            echo "Tree size $ts">>${datename}-update-${syn}-${rebuild}-result.$tree_name

            update_time=0
            N=2
            for((i=0;i<$N;i++))
            do 
                j=0
                #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 23 > tmp
                ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_uniform.txt 39 ../data_set/update_4096K ${syn} ${rebuild}> tmp
                cat tmp >>${datename}-update-${syn}-${rebuild}-result.$tree_name
     
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
done 
