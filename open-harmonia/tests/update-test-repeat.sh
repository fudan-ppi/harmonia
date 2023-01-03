#!/bin/bash
#rm *.hb+
tree_size=(23 24 25 26)
#update_batchs=(4096K 16K 32K 64K 128K 256K 512K 1024K 2048K 8192K 16384K)
update_batchs=(512K 4096K)
repeat=(90)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="hb+"
for ub in ${update_batchs[*]}
do
    for re in ${repeat[*]}
    do 
     
        echo "+++++++++++++$ub+repeat+$re+++++++++++++"
        for ts in ${tree_size[*]}
        do 
            echo -e "\n"
            echo "Tree size $ts"
            echo "=======================================================">>${datename}-update-${ub}-repeat${re}-result.$tree_name
            echo "Tree size $ts">>${datename}-update-${ub}-repeat${re}-result.$tree_name

            update_time=0
            N=2
            for((i=0;i<$N;i++))
            do 
                j=0
                #LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ../bpt_test ./insert_$ts.txt ./search_${distr}.txt 23 > tmp
                ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_uniform.txt 12 ../data_set/update_${ub}_repeat${re}> tmp
                cat tmp >>${datename}-update-${ub}-repeat${re}-result.$tree_name
     
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
            echo "${ts}-${ub}-repeate-${re}=========Result"
            echo "0$update_time"
         
        done
    done
done 
