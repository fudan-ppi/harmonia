
#ifndef CPU_SEARCH_H 
#define CPU_SEARCH_H
#include "bpt.h"
struct rightshift {
    inline key_t operator()(key_t x, unsigned offset) {return x>> offset;};
};
void search_cpu_prefetch(std::ifstream &file, BPlusTree &tree);
void search_cpu_prefetch_NTG(std::ifstream &file, BPlusTree &tree);
void search_cpu_prefetch_sort(std::ifstream &file, BPlusTree &tree);
void search_cpu_prefetch_sort_NTG(std::ifstream &file, BPlusTree &tree);

void search_range_cpu_prefetch_sort_NTG(std::ifstream &file, BPlusTree &tree);
void search_range_cpu_prefetch_HB(std::ifstream &file, BPlusTree &tree);


//avx function
//using in each file, not in main.cpp
void search_cpu_prefetch_HB(key_t* keys, value_t* vals, BPlusTree &tree);
void search_cpu_prefetch_PPI(key_t * keys, value_t* vals, BPlusTree &tree);
void search_cpu_prefetch_NTG_HB(key_t* keys, value_t* vals, BPlusTree &tree);
void search_cpu_prefetch_NTG_PPI(key_t * keys, value_t* vals, BPlusTree &tree);
void search_cpu_prefetch_NTG_PPI_wo_leaf(key_t * keys, int* leafnode_id, int* relist_idx, BPlusTree &tree);
void search_cpu_prefetch_HB_wo_leaf(key_t* keys,BLeaf_node ** bleafs,int* relist_idx, BPlusTree &tree);


void search_cpu_prefetch_NTG_HB_profile(key_t* keys, value_t* vals, BPlusTree &tree, long long &cmp_times_fact, long long &cmp_times_ideal);
void search_cpu_prefetch_NTG_PPI_profile(key_t * keys, value_t* vals, BPlusTree &tree, long long &cmp_times_fact, long long &cmp_times_ideal);



typedef struct{
    key_t key;
    value_t val;
}result_t;
void traverse_PPItree(key_t *start, key_t *end, int *leafnode_id, int *relist_idx, result_t (* values_range)[8], BPlusTree &tree);
void traverse_HB(key_t *start, key_t *end,BLeaf_node** bleaf_nodes,int *relist_idx, result_t(* values_range)[8]);


#endif
