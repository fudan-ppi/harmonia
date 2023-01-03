#include "helper.h"
#include "../src/cuda_utils.h"
void * transferToGPU(void *h,unsigned int size_byte){
    assert(h != NULL);/*{{{*/
    int *d = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&d,size_byte));
    CUDA_ERROR_HANDLER(cudaMemcpy(d,h,size_byte,cudaMemcpyHostToDevice));
    return d;/*}}}*/
}


void prepareGPU(CFB &tree, kArr_t* &d_node, int *&d_prefix, cArr_t* &d_leaf_record){
    
    
    
    int *prefixArray = tree.prefix;
    d_prefix = (int*)transferToGPU(prefixArray, tree.internal_node_num * sizeof(int));

    kArr_t *h_node = tree.key_section->getStart();
    d_node = (kArr_t *)transferToGPU(h_node, tree.node_num * sizeof(kArr_t));

    cArr_t *h_leaf_record = tree.pointer_section->elementAtIdx(tree.internal_node_num);
    d_leaf_record = (cArr_t *)transferToGPU(h_leaf_record, (tree.node_num-tree.internal_node_num)* sizeof(cArr_t));




}

void prepareGPU_noprefix(CFB &tree, kArr_t * &d_node, cArr_t * &d_child){
    
    
    kArr_t *h_node = tree.key_section->getStart();
    d_node = (kArr_t *)transferToGPU(h_node, tree.node_num * sizeof(kArr_t));

    cArr_t *h_child = tree.pointer_section->getStart();
    d_child = (cArr_t *)transferToGPU(h_child, tree.node_num * sizeof(cArr_t));

}


void prepareGPU_noprefix_RB(RB &tree, kArr_t * &d_node, cArr_t * &d_child){
    
    
    kArr_t *h_node = tree.key_section->getStart();
    d_node = (kArr_t *)transferToGPU(h_node, tree.node_num * sizeof(kArr_t));

    cArr_t *h_child = tree.pointer_section->getStart();
    d_child = (cArr_t *)transferToGPU(h_child, tree.node_num * sizeof(cArr_t));

}
