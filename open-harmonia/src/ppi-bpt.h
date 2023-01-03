#include"gbpt.h"
#include<fstream>

void PPI_BPT_Search_GPU_basic(BPlusTree &tree, std::ifstream &search_file);
void PPI_BPT_Search_GPU_basic_v2(BPlusTree &tree, std::ifstream &search_file);

void PPI_BPT_Search_GPU_V1_batch(BPlusTree &tree,std::ifstream &search_file);

void PPI_BPT_Search_GPU_V2_8thread(BPlusTree &tree, std::ifstream &search_file);
void PPI_BPT_Search_GPU_V2_8thread(BPlusTree &tree, std::ifstream &search_file, int startBit, int endBit);
void PPI_BPT_Search_GPU_V2_4thread(BPlusTree &tree, std::ifstream &search_file);

void PPI_BPT_Search_GPU_V3_2thread(BPlusTree &tree,std::ifstream &search_file);

void PPI_BPT_Search_GPU_V4_2thread(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V4_2thread_whole(BPlusTree &tree,std::ifstream &search_file);
#ifdef BOOST
void PPI_BPT_Search_GPU_V4_2thread_whole_cpusort(BPlusTree &tree,std::ifstream &search_file);
#endif
void PPI_BPT_Search_GPU_V5_2thread_new_tree(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V5_2thread_new_tree_CPUMultiThread(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V5_2thread_new_tree_serial(BPlusTree &tree,std::ifstream &search_file);


void PPI_BPT_Search_GPU_V6_balance(BPlusTree &tree, std::ifstream &search_file);
void PPI_BPT_Search_GPU_V7_2thread_new_tree_CPUMultiThread(BPlusTree &tree,std::ifstream &search_file);

void PPI_BPT_Search_GPU_V8_thread_scheduling(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V9_thread_scheduling(BPlusTree &tree,std::ifstream &search_file);


void PPI_BPT_Search_GPU_V10_multigpu_basedv5(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V11_multigpu_basedv5_cache(BPlusTree &tree,std::ifstream &search_file);//gpu cache

#ifdef BOOST
void PPI_BPT_Search_GPU_V12_thread_scheduling_cpu_cache(BPlusTree &tree,std::ifstream &search_file);

void PPI_BPT_Search_GPU_V13_thread_scheduling_cpu_cache_profile_version(BPlusTree &tree,std::ifstream &search_file);
#endif
void PPI_BPT_Search_GPU_V14_1_prefix_up_3_level_constant_down_readonly(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V14_2_prefix_up_full_constant_down_readonly(BPlusTree &tree,std::ifstream &search_file);
void PPI_BPT_Search_GPU_V14_3_prefix_opt(BPlusTree &tree,std::ifstream &search_file);


void test_prefix_2thread(BPlusTree &tree,std::ifstream &search_file);
void test_prefix_8thread(BPlusTree &tree,std::ifstream &search_file);
void test_prefix_sort_8thread(BPlusTree &tree,std::ifstream &search_file);
void test_sort_8thread(BPlusTree &tree,std::ifstream &search_file);
void test_8thread(BPlusTree &tree,std::ifstream &search_file);
void test_2thread(BPlusTree &tree,std::ifstream &search_file);
void test_sort_2thread(BPlusTree &tree,std::ifstream &search_file);


void compare_times_2thread(BPlusTree &tree,std::ifstream &search_file);
void compare_times_1thread(BPlusTree &tree,std::ifstream &search_file);
void compare_times_4thread(BPlusTree &tree,std::ifstream &search_file);


void multi_prefix_sort_1thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_sort_2thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_sort_4thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_sort_8thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_nosort_1thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_nosort_2thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_nosort_4thread(BPlusTree &tree,std::ifstream &search_file);
void multi_prefix_nosort_8thread(BPlusTree &tree,std::ifstream &search_file);
void multi_noprefix_nosort_8thread(BPlusTree &tree,std::ifstream &search_file);




//full
void full_multi_prefix_sort_1thread(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_nosort_1thread(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_sort_8thread(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_nosort_8thread(BPlusTree &tree,std::ifstream &search_file);

void full_multi_prefix_sort_1thread_notransfer(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_nosort_1thread_notransfer(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_sort_8thread_notransfer(BPlusTree &tree,std::ifstream &search_file);
void full_multi_prefix_nosort_8thread_notransfer(BPlusTree &tree,std::ifstream &search_file);

//bit test 
void bit_multi_prefix_sort_8thread(BPlusTree &tree,std::ifstream &search_file, int start_bit1, int end_bit1);
void bit_multi_prefix_sort_1thread(BPlusTree &tree,std::ifstream &search_file, int start_bit1, int end_bit1);



//no leaf test for batch update 
void multi_prefix_sort_1thread_noleaf(BPlusTree &tree,std::ifstream &search_file);




void range_prefix_sort_1thread_search_and_1thread_scan(BPlusTree &tree,std::ifstream &search_file);
void range_prefix_sort_1thread_search_and_1thread_scan_notransfer(BPlusTree &tree,std::ifstream &search_file);
void range_hb(BPlusTree &tree, std::ifstream &search_file);
void range_hb_notransfer(BPlusTree &tree, std::ifstream &search_file);

//V1 
//1. 8 thread , 中间sort, batch V2 ,key_back

//V2 
//1. 8 thread  sort first, batch V2,whole, key_pos_back ->blfnode 
//2. 4 thread  sort first, batch V2,whole, key_pos_back  ->blfnode

//v3
//1. 2 thread  sort first, batch V2,whole, key_pos_back -> v5 serial 

//void PPI_BPT_Search_GPU_v1_batchv1(BPlusTree &tree,std::ifstream &search_file);// @
//void PPI_BPT_Search_GPU_v1_batch_blocksort(BPlusTree &tree,std::ifstream &search_file);// @
//void PPI_BPT_Search_GPU_v1_batch_cache(BPlusTree &tree,std::ifstream &search_file);// @
//void PPI_BPT_Search_GPU_V2_4thread_pos_back(BPlusTree &tree, std::ifstream &search_file);// ！
//void PPI_BPT_Search_GPU_V3_2thread_whole(BPlusTree &tree,std::ifstream &search_file);// ->V5 ->serial !
//
//




void stick_this_thread_to_core(int core_id);
