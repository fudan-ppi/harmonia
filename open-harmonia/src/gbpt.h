#include"bpt.h"
#include<fstream>

Inner_node *prepareGPU(BPlusTree &tree);
Inner_node *prepareGPU_v1(BPlusTree &tree);
void prepareGPU_v2(BPlusTree &tree, Inner_node * &d_innode, int* &d_prefix);
void prepareGPU_leaf(BPlusTree &tree, BLeaf_node * &d_bleafnode);
void prepareGPU_leaf2(BPlusTree &tree, BLeaf_node * &d_bleafnode);

void BPT_Search_GPU(BPlusTree &tree,std::ifstream &search_file);
void BPT_Search_GPU_DoubleBuffering(BPlusTree &tree,std::ifstream &search_file);
void BPT_Search_GPU_DoubleBuffering_v2(BPlusTree &tree,std::ifstream &search_file);
void BPT_Search_GPU_DoubleBuffering_v3(BPlusTree &tree,std::ifstream &search_file);
void BPT_Search_GPU_multi_gpu_v4(BPlusTree &tree, std::ifstream &search_file);
void HB_simple(BPlusTree &tree, std::ifstream &search_file);
void HB_simple_notransfer(BPlusTree &tree, std::ifstream &search_file);


Inner_node * transferInnerToGPU(Inner_node *h_innode,unsigned int innodeSeg_size_byte);
int * transferArrayToGPU(int *h_array,unsigned int size_byte);
