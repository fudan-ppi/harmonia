#/bin/bash
#tree_size=(23)
tree_size=(23 24 25 26)
distribution=(uniform)
#start_bits=(0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)
start_bits=(63)
#rm *_result
for distr in ${distribution[*]}
do 
    echo "+++++++++++++$distr++++++++++++++"
    for ts in ${tree_size[*]}
    do 
        echo "Tree size $ts"
        for start_bit in ${start_bits[*]}
        do

            sort_time=0
            search_time=0
            gpu_time=0
                
            j=0
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 33 ${start_bit} 64 > tmp

            for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
            do  
                if [ $j -eq 0 ]
                then 
                    gpu_time=$(echo "scale=6; $a" | bc)
                fi 
                if [ $j -eq 1 ]
                then 
                    sort_time=$(echo "scale=6; $a" | bc)
                fi 
                if [ $j -eq 2 ]
                then 
                    search_time=$(echo "scale=6; $a" | bc)
                fi 
     
                j=$[ $j + 1 ]
            done 
             

            echo "${distr}-${ts}-${start_bit}-64=========Result gpu time; sort time; search time"
            echo "0$gpu_time, 0$sort_time, 0$search_time"
            
        done
    done
done
