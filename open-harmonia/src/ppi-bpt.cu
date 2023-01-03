#include<cuda_runtime.h>
#include<iostream>
#include<sys/time.h>
#include <assert.h>

#include <string>
#include"ppi-bpt.h"
#include"cuda_utils.h"
#include"mempool.h"

#include <fstream>
#include <omp.h>
#include "cub/cub.cuh"

#define PPI_Thread_Per_Query 8
#define M 1000000
#define Keys_Count 100*M
#define PPI_Block_Dim 512
#define PPI_Block_Per_SM 1024
#define KERNEL1_HEIGHT 3

typedef struct{
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;


using namespace std;

__global__ void ppi_bpt_search_kernel_up(Inner_node *d_innode, int root_idx, key_t *d_keys, int tree_height, int *d_inter_result,int *d_inter_idx, int key_count){
/*{{{*/
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ PPI_Thread_Per_Query;    
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = root_idx;
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target = d_keys[key_idx];

    for(int i=0;i<tree_height;i++){
        Inner_node *node = d_innode + start_idx[r];
        //search index;
        key_t idx_key = node->inner_index[search_idx];
        
        if(target < idx_key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
        
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            inner_index_result[r] = search_idx; 
        }
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
//-----------------------------------------------------------------------------------------
        //search key;
        int begin = inner_index_result[r]*8;
        int idx = begin + search_idx;
        key_t key = node->inner_key[idx];

        //===== shit
        if(search_idx == 0){
            start_idx[r] = (long)node->child[begin+8];
        }
        //=====

        if(target < key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
 
        
        //get next child;
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            start_idx[r]  = (long )node->child[idx]; 
        }
        inner_index_result[r] = 0;
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
    }
    d_inter_result[key_idx] = start_idx[r];
    d_inter_idx[key_idx] = key_idx;
    /*}}}*/
}



__global__ void ppi_bpt_search_kernel_down(Inner_node *d_innode ,int* d_inter_result, int *d_inter_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ PPI_Thread_Per_Query;
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = d_inter_result[key_idx];
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target = d_keys[d_inter_idx[key_idx]];

    GPU_Result &result = d_gresult[d_inter_idx[key_idx]];/*make g_result have the same order of d_keys*/

    for(int i=1;i<tree_height;i++){
        Inner_node *node = d_innode + start_idx[r];
        //search index;
        key_t idx_key = node->inner_index[search_idx];
        
        if(target < idx_key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
        
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            inner_index_result[r] = search_idx; 
        }
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
//-----------------------------------------------------------------------------------------
        //search key;
        int begin = inner_index_result[r]*8;
        int idx = begin + search_idx;
        key_t key = node->inner_key[idx];

        //===== shit
        if(search_idx == 0){
            if(i == tree_height-1){
                result.blfnode = (BLeaf_node *)node->child[0];
                result.relist_idx = begin+8;
            }else
                start_idx[r] = (long)node->child[begin+8];
        }
        //=====

        if(target < key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
 
        
        //get next child;
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            if(i==tree_height-1){
                result.blfnode = (BLeaf_node *)node->child[0];
                result.relist_idx = idx;
                return;
            }else 
                start_idx[r]  = (long )node->child[idx]; 
        }
        inner_index_result[r] = 0;
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
    }/*}}}*/
}


void PPI_BPT_Search_GPU_basic(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    Inner_node *d_innode = prepareGPU(tree);

    int rootIdx = tree.getRootIdx();

    assert(rootIdx != -1);
    int Thread_Per_Block = PPI_Block_Dim;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM;
    int Para_Search_Bucket = Block_Per_Grid * Thread_Per_Block / PPI_Thread_Per_Query;
    
    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_inter_result_size = sizeof(int) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    
    
    //host_malloc
    //key_t host_keys[Keys_Count];
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t));
    
    GPU_Result  *h_gresult = (GPU_Result *)malloc(batch_gresult_size);

    string s;
    int nums = 0;

    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
        //cout<<key<<endl;
    }


    //gpu_malloc
    key_t *d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size));
    
    int *d_inter_result_before;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_result_before, batch_inter_result_size));

    int *d_inter_idx_before;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_idx_before, batch_inter_result_size));

    int *d_inter_result_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_result_after, batch_inter_result_size));

    int *d_inter_idx_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_idx_after, batch_inter_result_size));
   
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size ));


    
    int kernel1_height = KERNEL1_HEIGHT;
    int kernel2_height = tree.getHeight() - kernel1_height;
    int total = nums / Para_Search_Bucket;


    value_t val[Para_Search_Bucket];
   


    struct timeval start;
    struct timeval end;
    double t_gpu_transfer_1 = 0;
    double t_gpu_up = 0;
    double t_gpu_sort = 0;
    double t_gpu_down = 0;
    double t_gpu_transfer_2 = 0;
    double t_cpu=0;



    for (int i=0;i<total;i++) {
        
        gettimeofday(&start, NULL); 
       
        
        CUDA_ERROR_HANDLER(cudaMemcpy(d_keys, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice));
        
        gettimeofday(&end, NULL); 
        
        t_gpu_transfer_1 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        gettimeofday(&start, NULL); 
       
        ppi_bpt_search_kernel_up<<<grid_dim, block_dim>>>(d_innode, rootIdx, d_keys, kernel1_height, d_inter_result_before, d_inter_idx_before, Para_Search_Bucket);

        cudaDeviceSynchronize();

        gettimeofday(&end, NULL); 
        t_gpu_up += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------
        
        gettimeofday(&start, NULL); 
        
        
        if (i==0) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_inter_idx_before, d_inter_idx_after, Para_Search_Bucket);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_inter_idx_before, d_inter_idx_after, Para_Search_Bucket);


        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        t_gpu_sort += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//-----------------------------------------------------------------------------        
        
        gettimeofday(&start, NULL); 
        
        ppi_bpt_search_kernel_down<<<grid_dim, block_dim>>>(d_innode, d_inter_result_after, d_inter_idx_after, d_keys, kernel2_height, d_gresult, Para_Search_Bucket);

        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
       
        t_gpu_down += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        
        gettimeofday(&start, NULL); 
        
        CUDA_ERROR_HANDLER(cudaMemcpy(h_gresult, d_gresult, batch_gresult_size, cudaMemcpyDeviceToHost));
  
        gettimeofday(&end, NULL);

        t_gpu_transfer_2 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------------------
        int ss = i * Para_Search_Bucket;

        gettimeofday(&start, NULL); 
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[ss + j];
            BLeaf_node *blfnode = h_gresult[j].blfnode;
            val[j] = blfnode->findKey(h_gresult[j].relist_idx, key);
        }
    
        gettimeofday(&end, NULL);
        t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        //test
       /*        
        for (int j=0;j<Para_Search_Bucket;j++) {
            key_t key = host_keys[ss+j];
            cout<<key<<": "<<val[j]<<endl;
        }
        */
    }


    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"t_gpu_transfer_1:    "<<t_gpu_transfer_1<<endl;
    cout<<"t_gpu_up:            "<<t_gpu_up<<endl;
    cout<<"t_gpu_sort:          "<<t_gpu_sort<<endl;
    cout<<"t_gpu_down:          "<<t_gpu_down<<endl;
    cout<<"t_gpu_transfer_2:    "<<t_gpu_transfer_2<<endl;
    cout<<"t_cpu:               "<<t_cpu<<endl;
    cout<<"t_gpu:               "<<t_gpu_up + t_gpu_down + t_gpu_sort << endl;
    cout<<"total time:          "<<t_gpu_up + t_gpu_down +t_gpu_transfer_1 +t_gpu_transfer_2+  t_gpu_sort + t_cpu<<endl;;
    

    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_result_before));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_result_after));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_idx_before));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_idx_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}

/*--------------------------------------------------------------------------
*
*   
*           PPI_GPU_V2
*           sort key and Inner_node result;
*
*
*-----------------------------------------------------------------------------*/



__global__ void ppi_bpt_search_kernel_up_v2(Inner_node *d_innode, int root_idx, key_t *d_keys, int tree_height, int *d_inter_result, int key_count){
/*{{{*/
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ PPI_Thread_Per_Query;    
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = root_idx;
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target = d_keys[key_idx];

    for(int i=0;i<tree_height;i++){
        Inner_node *node = d_innode + start_idx[r];
        //search index;
        key_t idx_key = node->inner_index[search_idx];
        
        if(target < idx_key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
        
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            inner_index_result[r] = search_idx; 
        }
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
//-----------------------------------------------------------------------------------------
        //search key;
        int begin = inner_index_result[r]*8;
        int idx = begin + search_idx;
        key_t key = node->inner_key[idx];

        //===== shit
        if(search_idx == 0){
            start_idx[r] = (long)node->child[begin+8];
        }
        //=====

        if(target < key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
 
        
        //get next child;
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            start_idx[r]  = (long )node->child[idx]; 
        }
        inner_index_result[r] = 0;
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
    }
    d_inter_result[key_idx] = start_idx[r];
    /*}}}*/
}





__global__ void ppi_bpt_search_kernel_down_v2(Inner_node *d_innode, int *d_inter_result,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ PPI_Thread_Per_Query;
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = d_inter_result[key_idx];
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target = d_keys[key_idx];

    GPU_Result &result = d_gresult[key_idx];/*g_result has a different order with d_keys before*/

    for(int i=1;i<tree_height;i++){
        Inner_node *node = d_innode + start_idx[r];
        //search index;
        key_t idx_key = node->inner_index[search_idx];
        
        if(target < idx_key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
        
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            inner_index_result[r] = search_idx; 
        }
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
//-----------------------------------------------------------------------------------------
        //search key;
        int begin = inner_index_result[r]*8;
        int idx = begin + search_idx;
        key_t key = node->inner_key[idx];

        //===== shit
        if(search_idx == 0){
            if(i == tree_height-1){
                result.blfnode = (BLeaf_node *)node->child[0];
                result.relist_idx = begin+8;
            }else
                start_idx[r] = (long)node->child[begin+8];
        }
        //=====

        if(target < key){
            flag[r][search_idx+1] = 1;
            selfFlag = 1;
        }
        __syncthreads();
 
        
        //get next child;
        if(selfFlag == 1 && flag[r][search_idx] == 0){
            if(i==tree_height-1){
                result.blfnode = (BLeaf_node *)node->child[0];
                result.relist_idx = idx;
                return;
            }else 
                start_idx[r]  = (long )node->child[idx]; 
        }
        inner_index_result[r] = 0;
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
    }/*}}}*/
}



void PPI_BPT_Search_GPU_basic_v2(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    Inner_node *d_innode = prepareGPU(tree);

    int rootIdx = tree.getRootIdx();

    assert(rootIdx != -1);
    int Thread_Per_Block = PPI_Block_Dim;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM;
    int Para_Search_Bucket = Block_Per_Grid * Thread_Per_Block / PPI_Thread_Per_Query;
    
    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_inter_result_size = sizeof(int) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    
    
    //host_malloc
    //key_t host_keys[Keys_Count];
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t));
    key_t *host_keys_after = (key_t *)malloc(batch_d_key_size);

    GPU_Result  *h_gresult = (GPU_Result *)malloc(batch_gresult_size);
    int *h_idx = (int *)malloc(batch_inter_result_size);
    string s;
    int nums = 0;

    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
        //cout<<key<<endl;
    }


    //gpu_malloc
    key_t *d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size));
    
    int *d_inter_result_before;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_result_before, batch_inter_result_size));

    int *d_inter_result_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_result_after, batch_inter_result_size));

    key_t *d_keys_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size));
   
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size ));


    
    int kernel1_height = KERNEL1_HEIGHT;
    int kernel2_height = tree.getHeight() - kernel1_height;
    int total = nums / Para_Search_Bucket;


    value_t val[Para_Search_Bucket];
   


    struct timeval start;
    struct timeval end;
    double t_gpu_up = 0;
    double t_gpu_transfer_1 = 0;
    double t_gpu_sort = 0;
    double t_gpu_down = 0;
    double t_gpu_transfer_2 = 0;
    double t_cpu=0;



    for (int i=0;i<total;i++) {
        
        gettimeofday(&start, NULL); 
       
        
        CUDA_ERROR_HANDLER(cudaMemcpy(d_keys, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice));
        
        gettimeofday(&end, NULL); 
        t_gpu_transfer_1 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
       
        gettimeofday(&start, NULL); 
        ppi_bpt_search_kernel_up_v2<<<grid_dim, block_dim>>>(d_innode, rootIdx, d_keys, kernel1_height, d_inter_result_before , Para_Search_Bucket);

        cudaDeviceSynchronize();

        gettimeofday(&end, NULL); 
        t_gpu_up += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------
        
        gettimeofday(&start, NULL); 
        
        
        if (i==0) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_keys, d_keys_after, Para_Search_Bucket);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_keys, d_keys_after, Para_Search_Bucket);


        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        t_gpu_sort += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//-----------------------------------------------------------------------------        
        
        gettimeofday(&start, NULL); 
        
        ppi_bpt_search_kernel_down_v2<<<grid_dim, block_dim>>>(d_innode, d_inter_result_after, d_keys_after, kernel2_height, d_gresult, Para_Search_Bucket);

        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);

        
        t_gpu_down += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

        gettimeofday(&start, NULL); 

        CUDA_ERROR_HANDLER(cudaMemcpy(h_gresult, d_gresult, batch_gresult_size, cudaMemcpyDeviceToHost));
         
        CUDA_ERROR_HANDLER(cudaMemcpy(host_keys_after, d_keys_after, batch_d_key_size, cudaMemcpyDeviceToHost));

        gettimeofday(&end, NULL);

        t_gpu_transfer_2 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------------------
        int ss = i * Para_Search_Bucket;

        gettimeofday(&start, NULL); 
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys_after[j];
            BLeaf_node *blfnode = h_gresult[j].blfnode;
            val[j] = blfnode->findKey(h_gresult[j].relist_idx, key);
        }
    
        gettimeofday(&end, NULL);
        t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        //test
        /*
        for (int j=0;j<Para_Search_Bucket;j++) {
            key_t key = host_keys_after[j];
            cout<<key<<": "<<val[j]<<endl;
        }
        */  
    }

    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"t_gpu_transfer_1:    "<<t_gpu_transfer_1<<endl;
    cout<<"t_gpu_up:            "<<t_gpu_up<<endl;
    cout<<"t_gpu_sort:          "<<t_gpu_sort<<endl;
    cout<<"t_gpu_down:          "<<t_gpu_down<<endl;
    cout<<"t_gpu_transfer_2:    "<<t_gpu_transfer_2<<endl;
    cout<<"t_cpu:               "<<t_cpu<<endl;
    cout<<"t_gpu:               "<<t_gpu_up + t_gpu_down + t_gpu_sort << endl;
    cout<<"total time:          "<<t_gpu_up + t_gpu_down +t_gpu_transfer_1 +t_gpu_transfer_2+  t_gpu_sort + t_cpu<<endl;;
    
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_result_before));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_result_after));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}


