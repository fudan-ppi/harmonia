#/bin/bash
tree_size=(23)
#tree_size=(24 25 26)
distribution=(uniform)
#start_bits=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64)
start_bits=(43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64)
#start_bits=(0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)
#start_bits=(0 60 64)
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
            ../bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 34 ${start_bit} 64 > tmp

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
