#!/bin/bash
# the default compiled bpt_test is for 64bit, switch to 32bit by update ./build.sh
dataset=./script/generate-data/dataset
default_insert_file=$dataset/insert64_4m/insert64_4m.txt
default_search_file=$dataset/search64_1m/search64_uniform_1m.txt
big_search_file=$dataset/search64_100m/search64_uniform_100m.txt
default_update_file=$dataset/update64_1024k/update64_7425_1024k.txt

option=0
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=1
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=2
echo $option
./bpt_test $default_insert_file $big_search_file $option

option=3
echo $option
./bpt_test $default_insert_file $big_search_file $option

option=5
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=6
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=11
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=12
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file

option=13
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file

option=16
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=17
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=18
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=19
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=20
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=21
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=22
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=23
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file

option=24
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=25
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=26
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=27
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=28
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=29
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=30
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=31
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=33
echo $option
./bpt_test $default_insert_file $default_search_file $option 63 64

option=34
echo $option
./bpt_test $default_insert_file $default_search_file $option 63 64

option=35
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=36
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=37
echo $option
./bpt_test $default_insert_file $default_search_file $option 63 64

option=38
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file

option=39
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file 64000 1048576

option=40
echo $option
./bpt_test $default_insert_file $default_search_file $option $default_update_file 64000

option=41
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=42
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=43
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=44
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=45
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=46
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=47
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=48
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=49
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=50
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=51
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=52
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=53
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=54
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=55
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=56
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=57
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=58
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=59
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=60
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=61
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=62
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=63
echo $option
./bpt_test $default_insert_file $default_search_file $option

option=64
echo $option
./bpt_test $default_insert_file $default_search_file $option


# ----------------------------------- time consuming ----------------------------------- 

# option=4
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=7
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=8
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=9
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=10
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=14
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

# option=15
# echo $option
# ./bpt_test $default_insert_file $big_search_file $option

