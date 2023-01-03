#!/bin/bash

echo link-to-64 

# insert 
ln -fs ../data_set/insert64/insert64_23.txt ../data_set/insert_23.txt  
ln -fs ../data_set/insert64/insert64_24.txt ../data_set/insert_24.txt  
ln -fs ../data_set/insert64/insert64_25.txt ../data_set/insert_25.txt  
ln -fs ../data_set/insert64/insert64_26.txt ../data_set/insert_26.txt 

#search 
ln -fs ../data_set/search64/search64_uniform.txt ../data_set/search_uniform.txt
ln -fs ../data_set/search64/search64_normal.txt ../data_set/search_normal.txt
ln -fs ../data_set/search64/search64_gamma.txt ../data_set/search_gamma.txt
ln -fs ../data_set/search64/search64_zipf.txt ../data_set/search_zipf.txt


#search_200M 
ln -fs ../data_set/search64_200M/search64_200M_uniform.txt ../data_set/search_200M_uniform.txt
ln -fs ../data_set/search64_200M/search64_200M_normal.txt ../data_set/search_200M_normal.txt
ln -fs ../data_set/search64_200M/search64_200M_gamma.txt ../data_set/search_200M_gamma.txt
#ln -fs ../data_set/search64_200M/search64_200M_zipf.txt ../data_set/search_200M_zipf.txt


#range 
ln -fs ../data_set/range64/range_uniform64.txt ../data_set/range_uniform.txt


#update 
ln -fs ../data_set/update64/update_4096K_repeat0 ../data_set/update_4096K_repeat0
ln -fs ../data_set/update64/update_4096K_repeat90 ../data_set/update_4096K_repeat90 
ln -fs ../data_set/update64/update_4096K_repeat95 ../data_set/update_4096K_repeat95 
ln -fs ../data_set/update64/update_4096K_repeat99 ../data_set/update_4096K_repeat99 
ln -fs ../data_set/update64/update_4096K_repeat999 ../data_set/update_4096K_repeat999 
ln -fs ../data_set/update64/update_4096K_repeat9999 ../data_set/update_4096K_repeat9999
ln -fs ../data_set/update64/update_4096K_repeat100 ../data_set/update_4096K_repeat100 

