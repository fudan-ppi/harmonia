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
#define PPI_Block_Dim 128
#define PPI_Block_Per_SM 64
#define KERNEL1_HEIGHT 2

//====batch 
#define GPU_SWP_SIZE 16
#define Sort_Per_Thread (GPU_SWP_SIZE / PPI_Thread_Per_Query)

typedef struct{
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;

using namespace std;
__global__ void ppi_bpt_search_kernel_V1_up(Inner_node *d_innode, int root_idx, key_t *d_keys, int tree_height, int *d_inter_result, int key_count){
/*{{{*/
    int key_base = PPI_Block_Dim / PPI_Thread_Per_Query * GPU_SWP_SIZE;
    int key_idx = key_base * blockIdx.x + threadIdx.x / PPI_Thread_Per_Query;    

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ int  start_idx[row_swp];
    

    int stride = PPI_Block_Dim / PPI_Thread_Per_Query;
    for(int k = 0;k < Sort_Per_Thread;k++){
        start_idx[threadIdx.x + k*blockDim.x]  =root_idx;
    }
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;

    for(int i=0;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE;j++){
            __syncthreads();
            int cur_r = r + stride *j;

            int cur_key = key_idx + stride * j ;
            if(cur_key>=key_count) continue;
            target = d_keys[cur_key];
            Inner_node *node = d_innode + start_idx[cur_r];
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
                start_idx[cur_r] = (long)node->child[begin+8];
            }
            //=====

            if(target < key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
            }
            __syncthreads();
     
            
            //get next child;
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                start_idx[cur_r]  = (long )node->child[idx]; 
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        }
        
    }
    if(search_idx == 0){
        for(int i=0;i<GPU_SWP_SIZE;i++){
            d_inter_result[key_idx + stride *i] = start_idx[r+ stride *i];
        }
    }
    /*}}}*/
}





__global__ void ppi_bpt_search_kernel_V1_down(Inner_node *d_innode, int *d_inter_result,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = PPI_Block_Dim / PPI_Thread_Per_Query * GPU_SWP_SIZE;
    int key_idx = key_base * blockIdx.x + threadIdx.x / PPI_Thread_Per_Query;    
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE;

    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ int  start_idx[row_swp];
    
    int stride = PPI_Block_Dim / PPI_Thread_Per_Query;
    
    for(int k = 0;k < Sort_Per_Thread;k++){
        int cur_key = key_base * blockIdx.x+ threadIdx.x+ k*blockDim.x;
        if(cur_key>= key_count) break;
        start_idx[threadIdx.x + k*blockDim.x] = d_inter_result[cur_key];
    }

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;


    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE;j++){
            __syncthreads();
            int cur_key = key_idx + stride *j;
            if(cur_key>=key_count) continue;

            int cur_r = r + stride *j;
            target =  d_keys[cur_key];
            GPU_Result &result = d_gresult[cur_key];

            Inner_node *node = d_innode + start_idx[cur_r];
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
                    start_idx[cur_r] = (long)node->child[begin+8];
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
                }else 
                    start_idx[cur_r]  = (long )node->child[idx]; 
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        }
    }/*}}}*/
}



void PPI_BPT_Search_GPU_V1_batch(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    Inner_node *d_innode = prepareGPU(tree);

    int rootIdx = tree.getRootIdx();

    assert(rootIdx != -1);

    int Thread_Per_Block = PPI_Block_Dim;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM;
    int Para_Search_Bucket = Block_Per_Grid * Thread_Per_Block / PPI_Thread_Per_Query * GPU_SWP_SIZE;
    
    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_inter_result_size = sizeof(int) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t));
    key_t *host_keys_after = (key_t *)malloc(batch_d_key_size);
    GPU_Result  *h_gresult = (GPU_Result *)malloc(batch_gresult_size);

    string s;
    int nums = 0;

    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
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
        ppi_bpt_search_kernel_V1_up<<<grid_dim, block_dim>>>(d_innode, rootIdx, d_keys, kernel1_height, d_inter_result_before , Para_Search_Bucket);

        cudaDeviceSynchronize();

        gettimeofday(&end, NULL); 
        t_gpu_up += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------
        
        gettimeofday(&start, NULL); 
        
        
        if (i==0) {// sort by inner_result
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_keys, d_keys_after, Para_Search_Bucket);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_inter_result_before, d_inter_result_after, d_keys, d_keys_after, Para_Search_Bucket);


        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        t_gpu_sort += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//-----------------------------------------------------------------------------        
        
        gettimeofday(&start, NULL); 
        
        ppi_bpt_search_kernel_V1_down<<<grid_dim, block_dim>>>(d_innode, d_inter_result_after, d_keys_after, kernel2_height, d_gresult, Para_Search_Bucket);

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
        
        for (int j=0;j<Para_Search_Bucket;j++) {
            key_t key = host_keys_after[j];
            cout<<key<<": "<<val[j]<<endl;
        }
        
    }

    cout<<"GPU Search V1[8 thread; up-down ; sort middle,key_back ; batch]"<<total * Para_Search_Bucket<<endl;
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
