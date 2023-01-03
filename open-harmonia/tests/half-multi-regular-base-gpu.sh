#!/bin/bash
tree_size=(23 24 25 26)
tree_order=(8 64)
narrow_rate=(1 2 4 8 16 32 64)
#narrow_rate=(16 32 64 128)
distribution=(uniform normal gamma zipf)
datename=$(date +%Y%m%d-%H%M%S)
tree_name="half-regular-base-gpu"
dir="${1}/regularB+"
echo $dir

for order in ${tree_order[*]}
do 
    for narrow in ${narrow_rate[*]}
    do
        if [ $order -lt $narrow ]
        then 
            echo "skip order:$order narrow:$narrow"
            continue 
        fi 
        cd ..
        cmake -DCMAKE_BUILD_TYPE=Release -DORDER=${order} -DNARROW=${narrow} . 2>&1 > /dev/null
        make 2>&1 > /dev/null 
        cd evaluation/

        for distr in ${distribution[*]}
        do 
            echo "+++++++++++++$distr++++++++++++++" >> ./$dir/result.${tree_name}_order${order}-narrow${narrow}
            for ts in ${tree_size[*]}
            do 
                echo "Start Order:${order}  Narrow:${narrow} Distribution:${distr} Tree—size:${ts}"
                echo -e "\n" >> ./$dir/result.${tree_name}_order${order}-narrow${narrow}
                echo "Tree size $ts" >> ./$dir/result.${tree_name}_order${order}-narrow${narrow}

                GPU1_time=0
                GPU2_time=0
                N=4
                for((i=0;i<$N;i++))
                do 
                    j=0
                    ../regular_bpt_test ../data_set/insert_$ts.txt ../data_set/search_${distr}.txt 11 > tmp
                    cat tmp >>${datename}-${distr}-order${order}-narrow${narrow}.$tree_name

                    for a in $(cat tmp | sed -n '/time/p' | gawk -F: '{printf "%f ",$2} END{prinf "\n"}')
                    do  
                        if [ $j -eq 0 ]
                        then 
                            echo "GPU1 + $a" >>  ./$dir/result.${tree_name}_order${order}-narrow${narrow}
                            GPU1_time=$(echo "scale=6; $GPU1_time + $a" | bc)
                        fi 
                        if [ $j -eq 1 ]
                        then 
                            echo "GPU2 + $a">>./$dir/result.${tree_name}_order${order}-narrow${narrow}

                            GPU2_time=$(echo "scale=6; $GPU2_time + $a" | bc)
                        fi 
                        j=$[ $j + 1 ]
                    done 
                done 
                echo $GPU1_time >>./$dir/result.${tree_name}_order${order}-narrow${narrow}
                echo $GPU2_time >>./$dir/result.${tree_name}_order${order}-narrow${narrow}

                GPU1_time=$(echo "scale=6;$GPU1_time / $N" | bc)
                GPU2_time=$(echo "scale=6;$GPU2_time / $N" | bc)
                echo "${distr}-${ts}=========Result">>./$dir/result.${tree_name}_order${order}-narrow${narrow}
                echo "0$GPU1_time,0$GPU2_time">>./$dir/result.${tree_name}_order${order}-narrow${narrow}
            done 
        done
        
               
    done
done 
echo "DONE!!!!"
