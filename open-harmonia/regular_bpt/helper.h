#include "cfb.h"
void prepareGPU(CFB &tree, kArr_t* &d_node, int* &d_prefix, cArr_t* &d_leaf_record);
void prepareGPU_noprefix(CFB &tree, kArr_t* &d_node, cArr_t* &d_child);
void prepareGPU_noprefix_RB(RB &tree, kArr_t* &d_node, cArr_t* &d_child);
