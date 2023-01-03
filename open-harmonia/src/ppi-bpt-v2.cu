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
#define M 1000000
#define Keys_Count 100*M

#define GPU_SWP_SIZE  16
#define PPI_Block_Dim 128
#define PPI_Block_Per_SM 64
#define PPI_Thread_Per_Query 8
#define  Sort_Per_Thread (GPU_SWP_SIZE / PPI_Thread_Per_Query)

#define GPU_SWP_SIZE4  16
#define PPI_Block_Dim_4thread 128
#define PPI_Block_Per_SM_4thread 64
#define PPI_Thread_Per_Query_4thread 4
#define  Sort_Per_4Thread (GPU_SWP_SIZE4 / PPI_Thread_Per_Query_4thread)

typedef struct{
    int blfnode;
    //BLeaf_node *blfnode;
    char relist_idx;
}GPU_Result;

using namespace std;

/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V2
*           using 8 thread
*           sort first
*           batch 
*           key pos back
*           whole
*
*-----------------------------------------------------------------------------*/

__global__ void ppi_bpt_V2_search_kernel_8threads(Inner_node *d_innode,  int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = PPI_Block_Dim / PPI_Thread_Per_Query * GPU_SWP_SIZE ;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim/ PPI_Thread_Per_Query;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE ;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    int stride = PPI_Block_Dim/ PPI_Thread_Per_Query;
    
    /*
    for(int i=0;i<GPU_SWP_SIZE2;i++){
        int cur_key = key_idx+stride*i;
        if(cur_key>=key_count)continue;
        start_idx[r + stride*i] = d_inter_result[cur_key];
    }
*/
 
    for (int k = 0; k<Sort_Per_Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;

    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE ;j++){
            __syncthreads();
            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;

            int cur_r = r+stride *j;

            target = d_keys[cur_key];
            GPU_Result &result=d_gresult[cur_key];

            int pos = start_idx[cur_r];
            Inner_node *node = d_innode + pos;
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
                    result.blfnode = (int)(BLeaf_node *)node->child[0];
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
                    result.relist_idx = idx ;
                    //return;
                }else
                    start_idx[cur_r] = (long)node->child[idx];
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        
        }
    }/*}}}*/
}

void PPI_BPT_Search_GPU_V2_8thread(BPlusTree &tree,ifstream &search_file, int startBit, int endBit){
/*{{{*/
    Inner_node *d_innode = prepareGPU_v1(tree);

    int rootIdx = tree.getRootIdx();
    
    
    assert(rootIdx != -1);
    int Thread_Per_Block = PPI_Block_Dim;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (13 * PPI_Block_Per_SM * Thread_Per_Block) / PPI_Thread_Per_Query*GPU_SWP_SIZE ;
    
    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;
    
    //host_malloc
    //key_t host_keys[Keys_Count];
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t));
    int *host_pos = (int *)malloc(batch_pos_size);

    GPU_Result  *h_gresult = (GPU_Result *)malloc(batch_gresult_size);
    string s;
    int nums = 0;

    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
        //cout<<key<<endl;
    }
     
    for (int i=0;i<Para_Search_Bucket;i++) {
        host_pos[i] = i;
    }


    //gpu_malloc
    key_t *d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size));
    

    key_t *d_keys_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size));
  
    int *d_pos;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos, batch_pos_size));

    int *d_pos_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos_after, batch_pos_size));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size ));


    
    int kernel_height = tree.getHeight() ;
    int total = nums / Para_Search_Bucket;


    value_t val[Para_Search_Bucket];
   


    struct timeval start;
    struct timeval end;
    double t_gpu_transfer_1 = 0;
    double t_gpu_sort = 0;
    double t_gpu_whole = 0;
    double t_gpu_transfer_2 = 0;
    double t_cpu=0;

    CUDA_ERROR_HANDLER(cudaMemcpy(d_pos, host_pos, batch_pos_size, cudaMemcpyHostToDevice));

    float time_gpu = 0;

    cudaEvent_t g_start;
    cudaEvent_t g_stop;
        
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    for (int i=0;i<total;i++) {

        
        
        cudaEventRecord(g_start);
        CUDA_ERROR_HANDLER(cudaMemcpy(d_keys, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice));
        cudaEventRecord(g_stop);
        cudaEventSynchronize(g_stop);
        
        cudaEventElapsedTime(&time_gpu,g_start,g_stop);
        

        t_gpu_transfer_1 += time_gpu/1000;
        //t_gpu_transfer_1 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------
        
        if (i==0) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos,d_pos_after,Para_Search_Bucket, startBit, endBit);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }

        cudaEventRecord(g_start);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos, d_pos_after, Para_Search_Bucket,startBit,endBit);
        cudaEventRecord(g_stop);
        cudaEventSynchronize(g_stop);

        cudaEventElapsedTime(&time_gpu,g_start,g_stop);


        t_gpu_sort += time_gpu/1000;
        //t_gpu_sort += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------

        
        cudaEventRecord(g_start);
        ppi_bpt_V2_search_kernel_8threads<<<grid_dim, block_dim>>>(d_innode, rootIdx , d_keys_after, kernel_height, d_gresult, Para_Search_Bucket);
        cudaEventRecord(g_stop);
        cudaEventSynchronize(g_stop);
        
        cudaEventElapsedTime(&time_gpu,g_start,g_stop);

        t_gpu_whole += time_gpu/1000;
        //t_gpu_whole += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

//---------------------------------------------------------------------

        cudaEventRecord(g_start);
        CUDA_ERROR_HANDLER(cudaMemcpy(h_gresult, d_gresult, batch_gresult_size, cudaMemcpyDeviceToHost));
         
        //CUDA_ERROR_HANDLER(cudaMemcpy(host_keys_after, d_keys_after, batch_d_key_size, cudaMemcpyDeviceToHost));
        CUDA_ERROR_HANDLER(cudaMemcpy(host_pos, d_pos_after, batch_pos_size, cudaMemcpyDeviceToHost));
        
        cudaEventRecord(g_stop);
        cudaEventSynchronize(g_stop);
        
        cudaEventElapsedTime(&time_gpu,g_start,g_stop);

        t_gpu_transfer_2 += time_gpu / 1000;
        //t_gpu_transfer_2 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------------------
        int ss = i * Para_Search_Bucket;

        gettimeofday(&start, NULL); 
       #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[host_pos[j]+ss];
            BLeaf_node *blfnode = tree.getLeafByIdx(h_gresult[j].blfnode);
            val[j] = blfnode->findKey(h_gresult[j].relist_idx, key);
        }
    
        gettimeofday(&end, NULL);
        t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        //test
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //     key_t key = host_keys[host_pos[j]+ss];
        //
        //    cout<<key<<": "<<val[j]<<endl;
        //}
    }

    //cout<<"GPU PPI V3 [sort key first, batch, 8threads,whole, pos_back, blfnode_int]"<<total * Para_Search_Bucket<<endl;
    //cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"t_gpu_transfer_1:    "<<t_gpu_transfer_1<<endl;
    cout<<"t_gpu_sort:          "<<t_gpu_sort<<endl;
    cout<<"t_gpu_whole:         "<<t_gpu_whole<<endl;
    cout<<"t_gpu_transfer_2:    "<<t_gpu_transfer_2<<endl;
    cout<<"t_cpu:               "<<t_cpu<<endl;
    cout<<"t_gpu:               "<<t_gpu_whole+ t_gpu_sort << endl;
    cout<<"total time:          "<<t_gpu_whole +t_gpu_transfer_1 +t_gpu_transfer_2+  t_gpu_sort + t_cpu<<endl;;
    
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}
void PPI_BPT_Search_GPU_V2_8thread(BPlusTree &tree, ifstream &search_file){
    PPI_BPT_Search_GPU_V2_8thread(tree, search_file, 48, 64);
}

/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V2
*           using 2 thread
*           batch
*           key pos sort
*           whole
*
*-----------------------------------------------------------------------------*/



__global__ void ppi_bpt_V2_search_kernel_4threads(Inner_node *d_innode,  int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = PPI_Block_Dim_4thread / PPI_Thread_Per_Query_4thread * GPU_SWP_SIZE4;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_4thread;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_4thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_4thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_4thread/ PPI_Thread_Per_Query_4thread;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE4;
    __shared__ char flag[row][5];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = PPI_Block_Dim_4thread / PPI_Thread_Per_Query_4thread;
    
   
    for (int k = 0; k<Sort_Per_4Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;

    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE4;j++){
            nexthalf[r] = -1;
            __syncthreads();
            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;

            int cur_r = r+stride *j;

            target = d_keys[cur_key];
            GPU_Result &result=d_gresult[cur_key];

            int pos = start_idx[cur_r];
            Inner_node *node = d_innode + pos;
            //search index;
            key_t idx_key = node->inner_index[search_idx];
            
            if(target < idx_key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = 0;
            }
            __syncthreads();
            
            if(nexthalf[r] == -1){
                idx_key = node->inner_index[search_idx+4];
                if(target < idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();
            }

            if(selfFlag == 1 && flag[r][search_idx] == 0){
                inner_index_result[r] = search_idx+nexthalf[r]*4; 
            }
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            nexthalf[r] = -1;
            __syncthreads();
    //-----------------------------------------------------------------------------------------
            //search key;
            int begin = inner_index_result[r]*8;
            int idx = begin + search_idx;
            key_t key = node->inner_key[idx];

            //===== shit
            if(search_idx == 0){
                if(i == tree_height-1){
                    result.blfnode = (int)(BLeaf_node *)node->child[0];
                    result.relist_idx = begin+8;
                }else
                    start_idx[cur_r] = (long)node->child[begin+8];
            }
            //=====

            if(target < key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = 0;
            }
            __syncthreads();
            if(nexthalf[r] == -1){
                key = node->inner_key[idx+4];
                if(target < key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();
            }
     
            
            //get next child;
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                if(i==tree_height-1){

                    result.relist_idx = idx + nexthalf[r]*4;
                    //return;
                }else
                    start_idx[cur_r] = (long)node->child[idx+nexthalf[r]*4];
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        
        }
    }/*}}}*/
}
void PPI_BPT_Search_GPU_V2_4thread(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    Inner_node *d_innode = prepareGPU_v1(tree);

    int rootIdx = tree.getRootIdx();
    
    
    assert(rootIdx != -1);
    int Thread_Per_Block = PPI_Block_Dim_4thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM_4thread ;
    int Para_Search_Bucket = (13 * PPI_Block_Per_SM_4thread * Thread_Per_Block) / PPI_Thread_Per_Query_4thread *GPU_SWP_SIZE4;
    
    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;
    
    
    //host_malloc
    //key_t host_keys[Keys_Count];
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t));
    int *host_pos = (int *)malloc(batch_pos_size);

    GPU_Result  *h_gresult = (GPU_Result *)malloc(batch_gresult_size);
    string s;
    int nums = 0;

    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
        //cout<<key<<endl;
    }
     
    for (int i=0;i<Para_Search_Bucket;i++) {
        host_pos[i] = i;
    }


    //gpu_malloc
    key_t *d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size));
    

    key_t *d_keys_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size));
  
    int *d_pos;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos, batch_pos_size));

    int *d_pos_after;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos_after, batch_pos_size));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size ));


    
    int kernel_height = tree.getHeight() ;
    int total = nums / Para_Search_Bucket;


    value_t val[Para_Search_Bucket];
   


    struct timeval start;
    struct timeval end;
    double t_gpu_transfer_1 = 0;
    double t_gpu_sort = 0;
    double t_gpu_whole = 0;
    double t_gpu_transfer_2 = 0;
    double t_cpu=0;

    CUDA_ERROR_HANDLER(cudaMemcpy(d_pos, host_pos, batch_pos_size, cudaMemcpyHostToDevice));


    for (int i=0;i<total;i++) {

        gettimeofday(&start, NULL); 
        
        CUDA_ERROR_HANDLER(cudaMemcpy(d_keys, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice));
        
        gettimeofday(&end, NULL); 
        t_gpu_transfer_1 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------
       gettimeofday(&start, NULL); 
        
        if (i==0) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos,d_pos_after,Para_Search_Bucket);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }
#ifdef TREE_32
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos, d_pos_after, Para_Search_Bucket,16,32);
#else
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos, d_pos_after, Para_Search_Bucket,48,64);
#endif
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        t_gpu_sort += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------

        gettimeofday(&start, NULL); 
        
        ppi_bpt_V2_search_kernel_4threads<<<grid_dim, block_dim>>>(d_innode, rootIdx , d_keys_after, kernel_height, d_gresult, Para_Search_Bucket);

        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        t_gpu_whole += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

        gettimeofday(&start, NULL); 
//---------------------------------------------------------------------

        CUDA_ERROR_HANDLER(cudaMemcpy(h_gresult, d_gresult, batch_gresult_size, cudaMemcpyDeviceToHost));
         
        //CUDA_ERROR_HANDLER(cudaMemcpy(host_keys_after, d_keys_after, batch_d_key_size, cudaMemcpyDeviceToHost));
        CUDA_ERROR_HANDLER(cudaMemcpy(host_pos, d_pos_after, batch_pos_size, cudaMemcpyDeviceToHost));

        gettimeofday(&end, NULL);

        t_gpu_transfer_2 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
//---------------------------------------------------------------------------------
        int ss = i * Para_Search_Bucket;

        gettimeofday(&start, NULL); 
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[host_pos[j]+ss];
            BLeaf_node *blfnode = tree.getLeafByIdx(h_gresult[j].blfnode);
            val[j] = blfnode->findKey(h_gresult[j].relist_idx, key);
        }
    
        gettimeofday(&end, NULL);
        t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        //test
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //     key_t key = host_keys[host_pos[j]+ss];
        //
        //    cout<<key<<": "<<val[j]<<endl;
        //}
    }

    cout<<"GPU PPI V3 [sort key first, batch, 4threads,whole, pos_back, blfnode_int]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"t_gpu_transfer_1:    "<<t_gpu_transfer_1<<endl;
    cout<<"t_gpu_sort:          "<<t_gpu_sort<<endl;
    cout<<"t_gpu_whole:         "<<t_gpu_whole<<endl;
    cout<<"t_gpu_transfer_2:    "<<t_gpu_transfer_2<<endl;
    cout<<"t_cpu:               "<<t_cpu<<endl;
    cout<<"t_gpu:               "<<t_gpu_whole+ t_gpu_sort << endl;
    cout<<"total time:          "<<t_gpu_whole +t_gpu_transfer_1 +t_gpu_transfer_2+  t_gpu_sort + t_cpu<<endl;;
    
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}


