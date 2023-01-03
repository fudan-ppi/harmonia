
#include<iostream>
#include<assert.h>
#include"gbpt.h"
#include"cuda_utils.h"
#include"mempool.h"
#include<fstream>
#include<string>
#include<sys/time.h>
using namespace std;
//----------------------------------------------------------------------------------Kernel
__global__ void InnerNodeAdressToIndex(Inner_node *d_innode,Inner_node *h_innode,unsigned int innodeSeg_size_byte,unsigned int innode_count){
    int innode_idx = blockIdx.x* (blockDim.x/BPT_ORDER) + threadIdx.x / BPT_ORDER;/*{{{*/
    Inner_node *innode = d_innode + innode_idx;
    int child_idx = threadIdx.x % BPT_ORDER;    
   
    if(innode_idx>=innode_count) return;
    if(!((char *)(innode->child[child_idx]) >= (char *)h_innode 
            && (char *)(innode->child[child_idx])< (char*)h_innode+innodeSeg_size_byte)) return;
    innode->child[child_idx] = (void *)((Inner_node *)innode->child[child_idx] -  h_innode);//NULL;/*}}}*/
}
__global__ void InnerNodeAdressToIndex_v1(Inner_node *d_innode,Inner_node *h_innode, BLeaf_node *h_bleafnode, unsigned int innodeSeg_size_byte,unsigned int leafSeg_size_byte, unsigned int innode_count){
    /*{{{*/
    int innode_idx = blockIdx.x* (blockDim.x/BPT_ORDER) + threadIdx.x / BPT_ORDER;
    Inner_node *innode = d_innode + innode_idx;
    int child_idx = threadIdx.x % BPT_ORDER;    
   
    if(innode_idx>=innode_count) return;
    if((char *)(innode->child[child_idx]) >= (char *)h_innode 
            && (char *)(innode->child[child_idx])< (char*)h_innode+innodeSeg_size_byte){
        
        innode->child[child_idx] = (void *)((Inner_node *)innode->child[child_idx] -  h_innode);
        return; 
    }
    if((char *)(innode->child[child_idx]) >= (char *)h_bleafnode 
            && (char *)(innode->child[child_idx])< (char*)h_bleafnode+leafSeg_size_byte){
        
        innode->child[child_idx] = (void *)((BLeaf_node *)innode->child[child_idx] -  h_bleafnode);
        return; 
    }

/*}}}*/
}

__global__ void BleafNodeAddressToIndex(BLeaf_node *d_bleafnode, BLeaf_node *h_bleafnode, unsigned int leafSeg_size_byte, unsigned int bleaf_count){
/*{{{*/
    int bleaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    BLeaf_node *bleaf = d_bleafnode + bleaf_idx;

    if (bleaf_idx >= bleaf_count) return;
    if ((char*)(bleaf->next) >= (char*)h_bleafnode 
            && (char *)(bleaf->next) < (char*)h_bleafnode + leafSeg_size_byte ) {
        bleaf->next = (BLeaf_node *)(void *)(bleaf->next - h_bleafnode);
    }
    else {
        bleaf->next = (BLeaf_node *)(-1);
    }
/*}}}*/
}
//---------------------------------------------------------------------------------- 
Inner_node * transferInnerToGPU(Inner_node *h_innode,unsigned int innodeSeg_size_byte){
    assert(h_innode != NULL);/*{{{*/
    Inner_node *d_innode = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_innode,innodeSeg_size_byte));
    CUDA_ERROR_HANDLER(cudaMemcpy(d_innode,h_innode,innodeSeg_size_byte,cudaMemcpyHostToDevice));
    return d_innode;/*}}}*/
}

int * transferArrayToGPU(int *h_array,unsigned int size_byte){
    assert(h_array != NULL);/*{{{*/
    int *d_array = NULL;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_array,size_byte));
    CUDA_ERROR_HANDLER(cudaMemcpy(d_array,h_array,size_byte,cudaMemcpyHostToDevice));
    return d_array;/*}}}*/
}

void * transferToGPU(void *h, unsigned int size_byte){
    
    int *d = NULL;/*{{{*/
    CUDA_ERROR_HANDLER(cudaMalloc(&d,size_byte));
    CUDA_ERROR_HANDLER(cudaMemcpy(d,h,size_byte,cudaMemcpyHostToDevice));
    return d;/*}}}*/

}

// prepareGPU:      old tree or new_tree    
//Transfer inner_node to GPU.   Inner_node_ptr in inner_node are offset on GPU. 
Inner_node *prepareGPU(BPlusTree &tree){
    Inner_node *h_innode =  NULL;/*{{{*/
    Inner_node *d_innode = NULL;
    unsigned int innodeSeg_size_byte = 0;

    tree.getInnerSegementationInfo(h_innode,innodeSeg_size_byte);
    d_innode = transferInnerToGPU(h_innode,innodeSeg_size_byte);
    //change the child address to index;
    unsigned int innode_count = innodeSeg_size_byte / sizeof(Inner_node); 

    int thread_pre_innode = BPT_ORDER;
    int thread_pre_block = 1024;
    int innode_pre_block = thread_pre_block/ thread_pre_innode;

    dim3 block_dim(1024);
    dim3 grid_dim((innode_count+innode_pre_block-1)/ innode_pre_block);

    InnerNodeAdressToIndex<<<grid_dim,block_dim>>>(d_innode,h_innode,innodeSeg_size_byte,innode_count);

    cudaDeviceSynchronize();
    return d_innode;/*}}}*/
}

// prepareGPU_v1:   old tree or new_tree     
//Transfer inner_node to GPU.   Inner_node_ptr and Leaf_node_ptr in inner_node are offset on GPU. 
Inner_node *prepareGPU_v1(BPlusTree &tree) {
    /*{{{*/
    Inner_node *h_innode =  tree.getInnerNodeStart();
    BLeaf_node *h_bleafnode = tree.getBLeafNodeStart();
    
    unsigned int innodeSeg_size_byte = tree.getInnerSize() * sizeof(Inner_node);
    unsigned int leafSeg_size_byte = tree.getBLeafSize() * sizeof(BLeaf_node);
    unsigned int innode_count = tree.getInnerSize();
    
    Inner_node *d_innode = transferInnerToGPU(h_innode,innodeSeg_size_byte);

    int thread_pre_innode = BPT_ORDER;
    int thread_pre_block = 1024;
    int innode_pre_block = thread_pre_block/ thread_pre_innode;

    dim3 block_dim(1024);
    dim3 grid_dim((innode_count+innode_pre_block-1)/ innode_pre_block);

    InnerNodeAdressToIndex_v1<<<grid_dim,block_dim>>>(d_innode,h_innode,h_bleafnode, innodeSeg_size_byte,leafSeg_size_byte,innode_count);

    cudaDeviceSynchronize();
    return d_innode;
/*}}}*/
}

// prepareGPU_v2:   new_tree   
//Transfer inner_node to GPU.
void prepareGPU_v2(BPlusTree &tree,  Inner_node * &d_innode, int * &d_prefix) {
    int *prefixArray = tree.getPrefixArray();/*{{{*/
    assert(prefixArray != NULL);

    Inner_node *h_innode =  NULL;
    
    unsigned int innodeSeg_size_byte = 0;

    tree.getInnerSegementationInfo(h_innode,innodeSeg_size_byte);
    d_innode = transferInnerToGPU(h_innode,innodeSeg_size_byte);
    d_prefix = transferArrayToGPU(prefixArray, tree.getInnerSize() * sizeof(int) );
/*}}}*/
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//prepareGPU_leaf   for full search
//Transfer leaf_node to GPU.
void prepareGPU_leaf(BPlusTree &tree, BLeaf_node *&d_bleafnode) {

    BLeaf_node *h_bleafnode = tree.getBLeafNodeStart(); 
    int size = tree.getBLeafSize()*sizeof(BLeaf_node);
    
    d_bleafnode = (BLeaf_node *)transferToGPU((BLeaf_node *)h_bleafnode, size);
}

//prepareGPU_leaf2   for range query
//Transfer leaf_node to GPU.  "Next" pointer in bleaf_node are offset on GPU.

void prepareGPU_leaf2(BPlusTree &tree, BLeaf_node *&d_bleafnode) {

    int bleaf_count = tree.getBLeafSize();
    BLeaf_node *h_bleafnode = tree.getBLeafNodeStart(); 
    int size = bleaf_count*sizeof(BLeaf_node);
    
    d_bleafnode = (BLeaf_node *)transferToGPU((BLeaf_node *)h_bleafnode, size);

    int thread_per_bleafnode = 1;  
    int thread_per_block = 1024;
    int bleafnode_per_block = thread_per_block / thread_per_bleafnode;

    dim3 block_dim(thread_per_block);
    dim3 grid_dim((bleaf_count + bleafnode_per_block-1)/bleafnode_per_block );

    BleafNodeAddressToIndex<<<grid_dim, block_dim>>>(d_bleafnode, h_bleafnode, size, bleaf_count);
    cudaDeviceSynchronize();
    
}
