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

#ifdef BOOST
#include <boost/sort/spreadsort/spreadsort.hpp>
#endif


#define M 1000000
#define Keys_Count 100*M

#define KERNEL1_HEIGHT 2
#define  GPU_SWP_SIZE2  16


#define PPI_Block_Dim_2thread 128
#define PPI_Block_Per_SM 64

#define PPI_Thread_Per_Query_2thread 2
#define  Sort_Per_2Thread (GPU_SWP_SIZE2 / PPI_Thread_Per_Query_2thread)

typedef struct{
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;


using namespace std;
#ifdef BOOST
using namespace boost::sort::spreadsort;
#endif


/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V4
*           using 2 thread
*           batch
            key pos sort
*
*
*-----------------------------------------------------------------------------*/
__global__ void ppi_bpt_V4_search_kernel_up_2threads(Inner_node *d_innode, int root_idx, key_t *d_keys, int tree_height, int *d_inter_result, int key_count){
/*{{{*/
    int key_base =  PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_2thread;  
    if(key_idx >= key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_2thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_2thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_2thread/ PPI_Thread_Per_Query_2thread;//blockDim.x/ PPI_Thread_Per_Query_2thread;
    const int row_swp = row * GPU_SWP_SIZE2;

    __shared__ char flag[row][3];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];
    
    __shared__ char nexthalf[row];
   


    int stride = PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread;
    
    for (int k = 0; k<Sort_Per_2Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }
    
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target ;
    key_t idx_key = 0;
    key_t key = 0;
    
    for(int i=0;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE2;j++){
            nexthalf[r] = -1;
            __syncthreads();
            int cur_r = r + stride * j;

            int cur_key = key_idx+stride*j;
            if(cur_key >= key_count) continue;

            target = d_keys[cur_key];

            int inner_node_idx = start_idx[cur_r];
            Inner_node *node = d_innode + inner_node_idx;
            //search index;
            idx_key = node->inner_index[search_idx];
            
            if(target < idx_key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = 0;
            }
            __syncthreads();

            if(nexthalf[r] == -1){
                idx_key = node->inner_index[search_idx+2];
                if(target < idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    idx_key = node->inner_index[search_idx+4];
                    if(target < idx_key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        idx_key = node->inner_index[search_idx+6];
                        if(target < idx_key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }
            
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                inner_index_result[r] = search_idx+nexthalf[r]*2; 
            }
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            nexthalf[r] = -1;
            __syncthreads();
    //-----------------------------------------------------------------------------------------
            //search key;
            int begin = inner_index_result[r]*8;
            int idx = begin + search_idx;

            key = node->inner_key[idx];

            //===== shit
            if(search_idx == 0){
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
                key = node->inner_key[idx+2];
                if(target < key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    key = node->inner_key[idx+4];
                    if(target < key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        key = node->inner_key[idx+6];
                        if(target < key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }
     
            
            //get next child;
            
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                start_idx[cur_r]  =(long)node->child[idx+nexthalf[r]*2]; 
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        }
    }
    if(search_idx == 0){
        for(int i=0;i<GPU_SWP_SIZE2;i++){
            if(start_idx[r+stride*i]<=0)
            d_inter_result[key_idx+stride*i] = -999999;
            else 
            d_inter_result[key_idx+stride*i] = start_idx[r+stride*i];
        }
    }
    /*}}}*/
}


__global__ void ppi_bpt_V4_search_kernel_down_2threads(Inner_node *d_innode, int *d_inter_result,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_2thread;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_2thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_2thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_2thread/ PPI_Thread_Per_Query_2thread;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE2;
    __shared__ char flag[row][3];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread;
    
    /*
    for(int i=0;i<GPU_SWP_SIZE2;i++){
        int cur_key = key_idx+stride*i;
        if(cur_key>=key_count)continue;
        start_idx[r + stride*i] = d_inter_result[cur_key];
    }
*/
    for (int k = 0; k<Sort_Per_2Thread; k++){

        int cur_key = key_base * blockIdx.x + threadIdx.x + k * blockDim.x;
        if(cur_key>=key_count)break;
        start_idx[threadIdx.x + k* blockDim.x] = d_inter_result[cur_key];
    }
           

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;

    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE2;j++){
            nexthalf[r] = -1;
            __syncthreads();
            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;

            int cur_r = r+stride *j;

            target = d_keys[cur_key];
            GPU_Result &result=d_gresult[cur_key];

            Inner_node *node = d_innode + start_idx[cur_r];
            //search index;
            key_t idx_key = node->inner_index[search_idx];
            
            if(target < idx_key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = 0;
            }
            __syncthreads();
            
            if(nexthalf[r] == -1){
                idx_key = node->inner_index[search_idx+2];
                if(target < idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    idx_key = node->inner_index[search_idx+4];
                    if(target < idx_key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        idx_key = node->inner_index[search_idx+6];
                        if(target < idx_key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }

            if(selfFlag == 1 && flag[r][search_idx] == 0){
                inner_index_result[r] = search_idx+nexthalf[r]*2; 
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
                    result.blfnode = (BLeaf_node *)node->child[0];
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
                key = node->inner_key[idx+2];
                if(target < key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    key = node->inner_key[idx+4];
                    if(target < key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        key = node->inner_key[idx+6];
                        if(target < key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }

     
            
            //get next child;
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                if(i==tree_height-1){
                    result.blfnode = (BLeaf_node *)node->child[0];
                    result.relist_idx = idx + nexthalf[r]*2;
                }else 
                    start_idx[cur_r]  = (long )node->child[idx+nexthalf[r]*2]; 
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        
        }
    }/*}}}*/
}

namespace{
    volatile int stream_flag[2] = {-1,-1};/*{{{*/
    int Thread_Per_Block = PPI_Block_Dim_2thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (Block_Per_Grid * Thread_Per_Block) / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;

    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_inter_result_size = sizeof(int) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;

    key_t *host_keys;
    int *host_pos; 
    GPU_Result  *h_gresult;


    key_t *d_keys;
    int *d_inter_result;
    key_t *d_keys_after;
    int *d_pos;
    int *d_pos_after;
    GPU_Result *d_gresult;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    int rootIdx; 

    int kernel1_height = KERNEL1_HEIGHT;
    int kernel2_height; 
    int total;

    Inner_node *d_innode; 
    cudaStream_t stream[2];


    void CUDART_CB CallBack(cudaStream_t stream,cudaError_t status,void *data){
        stream_flag[(size_t)data%2] = (size_t)data;
    }/*}}}*/

    float time_gpu = 0;

    cudaEvent_t g_start,g_stop;

    volatile int key_status = -1;
}


void* launch_kernel_thread(void *args){
    for(int i=0;i<total;i++){/*{{{*/
        int idx = i%2;
        int stride = idx*Para_Search_Bucket;
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + stride, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[idx]));
        const key_t *tmp_dkeys =d_keys+stride;
        const int *tmp_dpos =d_pos+stride;
        
#ifdef TREE_32
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,48,64,stream[idx]);
#else 
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,16,32,stream[idx]);
#endif  

        ppi_bpt_V4_search_kernel_up_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, rootIdx, d_keys_after+stride, kernel1_height, d_inter_result+stride, Para_Search_Bucket);

        ppi_bpt_V4_search_kernel_down_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, d_inter_result+stride, d_keys_after+stride, kernel2_height, d_gresult+stride, Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);
    }
    return NULL;/*}}}*/
}


void* launch_kernel_thread_measure(void *args){

/*{{{*/


    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start, stream[1]);
    struct timeval start;
    struct timeval end;

    //gettimeofday(&start, NULL); 
    for(int i=0;i<total;i++){
        int idx = i%2;
        int stride = idx*Para_Search_Bucket;
        
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + stride, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[idx]));
        const key_t *tmp_dkeys =d_keys+stride;
        const int *tmp_dpos =d_pos+stride;
#ifdef TREE_32
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,16,32,stream[idx]);
#else 
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,48,64,stream[idx]);
#endif
        ppi_bpt_V4_search_kernel_up_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, rootIdx, d_keys_after+stride, kernel1_height, d_inter_result+stride, Para_Search_Bucket);

        ppi_bpt_V4_search_kernel_down_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, d_inter_result+stride, d_keys_after+stride, kernel2_height, d_gresult+stride, Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);


    }
    //cudaDeviceSynchronize();
    //gettimeofday(&end, NULL); 
    //time_gpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

    cudaEventRecord(g_stop, stream[1]);
   
    /*for (int i=0 ;i<2; i++) {
        cudaStreamDestroy(stream[i]);
    }*/
    return NULL;/*}}}*/
}


void PPI_BPT_Search_GPU_V4_2thread(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    d_innode = prepareGPU(tree);

    rootIdx = tree.getRootIdx();
    kernel2_height = tree.getHeight() - kernel1_height;

    assert(rootIdx != -1);
    
    //host_malloc
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));
   
    //init key 
    int nums = 0;
    string s;
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    } 
    total = nums / Para_Search_Bucket;
    
    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*nums));
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_pos,sizeof(int)*nums));

    //init key pos
    for (int i=0;i<Para_Search_Bucket;i++) {
        host_pos[i] = i;
        host_pos[i+Para_Search_Bucket] = i;
    }
    for(int i=0;i<2;i++) cudaStreamCreate(&stream[i]);

    //gpu_malloc
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_inter_result, batch_inter_result_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size*2));
  
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos, batch_pos_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos_after, batch_pos_size*2));

    
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size *2));

        
    value_t val[Para_Search_Bucket];

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time=0;
    double cpu_time = 0;

    CUDA_ERROR_HANDLER(cudaMemcpy(d_pos, host_pos, batch_pos_size*2, cudaMemcpyHostToDevice));
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos,d_pos_after,Para_Search_Bucket,16,32);
    CUDA_ERROR_HANDLER(cudaMalloc(&d_temp_storage, temp_storage_bytes*2));

    pthread_t tid;
    //if((pthread_create(&tid,NULL,launch_kernel_thread,NULL))!=0){
    if((pthread_create(&tid,NULL,launch_kernel_thread_measure,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }

    gettimeofday(&start, NULL); 
    for (int i=0;i<total;i++) {
        int idx = i%2;

        int start = i * Para_Search_Bucket;

        while(stream_flag[idx]<i);


        gettimeofday(&start1, NULL); 
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[start + host_pos[start+j]];
            BLeaf_node *blfnode = h_gresult[start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[start+j].relist_idx, key);
        }
    
        gettimeofday(&end1, NULL);

        cpu_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
       //test
        
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //    key_t key = host_keys[start + host_pos[start+j]];
        //    cout<<key<<": "<<val[j]<<endl;
        //}
       
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"GPU PPI V4 [sort key first,2threads,doublebuffer]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;

    cout<<"CPU time                 "<<cpu_time<<endl;
    cout<<"GPU time(one stream)     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;
    
    
    CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_pos));
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_inter_result));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}


/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V4
*           using 2 thread
*           batch
*           key pos sort
*           whole
*
*
*-----------------------------------------------------------------------------*/

__global__ void ppi_bpt_V4_search_kernel_2threads(Inner_node *d_innode, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_2thread;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_2thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_2thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_2thread/ PPI_Thread_Per_Query_2thread;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE2;
    __shared__ char flag[row][3];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = PPI_Block_Dim_2thread / PPI_Thread_Per_Query_2thread;
    
    /*
    for(int i=0;i<GPU_SWP_SIZE2;i++){
        int cur_key = key_idx+stride*i;
        if(cur_key>=key_count)continue;
        start_idx[r + stride*i] = d_inter_result[cur_key];
    }
*/
 
    for (int k = 0; k<Sort_Per_2Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;

    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE2;j++){
            nexthalf[r] = -1;
            __syncthreads();
            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;

            int cur_r = r+stride *j;

            target = d_keys[cur_key];
            GPU_Result &result=d_gresult[cur_key];

            Inner_node *node = d_innode + start_idx[cur_r];
            //search index;
            key_t idx_key = node->inner_index[search_idx];
            
            if(target < idx_key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = 0;
            }
            __syncthreads();
            
            if(nexthalf[r] == -1){
                idx_key = node->inner_index[search_idx+2];
                if(target < idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    idx_key = node->inner_index[search_idx+4];
                    if(target < idx_key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        idx_key = node->inner_index[search_idx+6];
                        if(target < idx_key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }

            if(selfFlag == 1 && flag[r][search_idx] == 0){
                inner_index_result[r] = search_idx+nexthalf[r]*2; 
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
                    result.blfnode = (BLeaf_node *)node->child[0];
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
                key = node->inner_key[idx+2];
                if(target < key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                __syncthreads();

                if(nexthalf[r] == -1){
                    key = node->inner_key[idx+4];
                    if(target < key){
                        flag[r][search_idx+1] = 1;
                        selfFlag = 1;
                        nexthalf[r] = 2;
                    }
                    __syncthreads();
                    if(nexthalf[r] == -1){
                        key = node->inner_key[idx+6];
                        if(target < key){
                            flag[r][search_idx+1] = 1;
                            selfFlag = 1;
                            nexthalf[r] = 3;
                        }
                        __syncthreads();
                    }
                }
            }

     
            
            //get next child;
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                if(i==tree_height-1){
                    result.blfnode = (BLeaf_node *)node->child[0];
                    result.relist_idx = idx + nexthalf[r]*2;
                    //return;
                }else 
                    start_idx[cur_r]  = (long )node->child[idx+nexthalf[r]*2]; 
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        
        }
    }/*}}}*/
}



void* launch_kernel_thread_whole_measure(void *args){

/*{{{*/
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);


    cudaEventRecord(g_start);
    for(int i=0;i<total;i++){
        int idx = i%2;
        int stride = idx*Para_Search_Bucket;
        const key_t *tmp_dkeys =d_keys+stride;
        const int *tmp_dpos =d_pos+stride;
        

        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + stride, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[idx]));

#ifdef TREE_32
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,16,32,stream[idx]);
#else 
        cub::DeviceRadixSort::SortPairs(d_temp_storage + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after+stride, tmp_dpos, d_pos_after+stride, Para_Search_Bucket,48,64,stream[idx]);
#endif
        ppi_bpt_V4_search_kernel_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, rootIdx, d_keys_after+stride, kernel2_height , d_gresult+stride, Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);


    }
    
    cudaEventRecord(g_stop);
   
       return NULL;/*}}}*/
}
void PPI_BPT_Search_GPU_V4_2thread_whole(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    d_innode = prepareGPU(tree);

    rootIdx = tree.getRootIdx();

    kernel2_height = tree.getHeight();
    assert(rootIdx != -1);
    
    //host_malloc
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));
   
    //init key 
    int nums = 0;
    string s;
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    } 
    total = nums / Para_Search_Bucket;
    
    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*nums));
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_pos,sizeof(int)*nums));

    //init key pos
    for (int i=0;i<Para_Search_Bucket;i++) {
        host_pos[i] = i;
        host_pos[i+Para_Search_Bucket] = i;
    }
    for(int i=0;i<2;i++) cudaStreamCreate(&stream[i]);

    //gpu_malloc
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size*2));
  
    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos, batch_pos_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_pos_after, batch_pos_size*2));

    
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size *2));

        
    value_t val[Para_Search_Bucket];

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
       double total_time=0;
    double cpu_time = 0;

    CUDA_ERROR_HANDLER(cudaMemcpy(d_pos, host_pos, batch_pos_size*2, cudaMemcpyHostToDevice));
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_after, d_pos,d_pos_after,Para_Search_Bucket,16,32);
    CUDA_ERROR_HANDLER(cudaMalloc(&d_temp_storage, temp_storage_bytes*2));

    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread_whole_measure,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }

    gettimeofday(&start, NULL); 
    for (int i=0;i<total;i++) {
        int idx = i%2;

        int start = i * Para_Search_Bucket;
        while(stream_flag[idx]<i);


        gettimeofday(&start1, NULL);
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[start + host_pos[start+j]];
            BLeaf_node *blfnode = h_gresult[start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[start+j].relist_idx, key);
        }
    
        gettimeofday(&end1, NULL);

        cpu_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
       //test
        
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //    key_t key = host_keys[start + host_pos[start+j]];
        //    cout<<key<<": "<<val[j]<<endl;
        //}
       
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"GPU PPI V4 [sort key first,2threads,doublebuffer, whole]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;

    cout<<"CPU time                 "<<cpu_time<<endl;
    cout<<"GPU time(one stream)     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;
    
    
    CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_pos));
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}



#ifdef BOOST
/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V4
*           using 2 thread
*           batch
*           key sort on CPU
*           whole
*
*
*-----------------------------------------------------------------------------*/
struct rightshift {
    inline int operator()(key_t x, unsigned offset) {return x>> offset;;;;;;;;;};
};

void *sort_key_CPU(void *arg){
    int i = 0;
 struct timeval start;
    struct timeval end;
    double total_time=0;
    gettimeofday(&start, NULL); 
    omp_set_num_threads(30);
#pragma omp parallel for 
    for(int i=0;i<total;i++){
        integer_sort(host_keys+i*Para_Search_Bucket, host_keys+(i+1)*Para_Search_Bucket,rightshift());
        key_status++;
    } 
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    cout<<"CPU sort time:    "<<total_time<<endl;

    return NULL;
}
void* launch_kernel_thread_whole_measure_no_sort(void *args){

/*{{{*/
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start, stream[1]);
    struct timeval start;
    struct timeval end;

    //gettimeofday(&start, NULL); 
    for(int i=0;i<total;i++){
        int idx = i%2;
        int stride = idx*Para_Search_Bucket;
       while(key_status<i);

        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys_after + stride, host_keys + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[idx]));

        ppi_bpt_V4_search_kernel_2threads<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode, rootIdx, d_keys_after+stride, kernel2_height , d_gresult+stride, Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);
    }
    //cudaDeviceSynchronize();
    //gettimeofday(&end, NULL); 
    //time_gpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

    cudaEventRecord(g_stop, stream[1]);
   
    /*for (int i=0 ;i<2; i++) {
        cudaStreamDestroy(stream[i]);
    }*/
    return NULL;/*}}}*/
}
void PPI_BPT_Search_GPU_V4_2thread_whole_cpusort(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    d_innode = prepareGPU(tree);

    rootIdx = tree.getRootIdx();

    kernel2_height = tree.getHeight();
    assert(rootIdx != -1);
    
    //host_malloc
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));
   
    //init key 
    int nums = 0;
    string s;
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    } 
    total = nums / Para_Search_Bucket;
    
    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*nums));

    //init key pos
    for(int i=0;i<2;i++) cudaStreamCreate(&stream[i]);

    //gpu_malloc
    //CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, batch_d_key_size*2));

    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_after, batch_d_key_size*2));
  
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult, batch_gresult_size *2));

        
    value_t val[Para_Search_Bucket];

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time=0;
    double cpu_time = 0;

    
    
    pthread_t tid;
    if((pthread_create(&tid,NULL,sort_key_CPU,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }

    pthread_t tid1;
    if((pthread_create(&tid1,NULL,launch_kernel_thread_whole_measure_no_sort,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }

    gettimeofday(&start, NULL); 
    for (int i=0;i<total;i++) {
        int idx = i%2;

        int start = i * Para_Search_Bucket;

        while(stream_flag[idx]<i);


        gettimeofday(&start1, NULL); 
        if(key_status <total) omp_set_num_threads(26);
        #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[start + j];
            BLeaf_node *blfnode = h_gresult[start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[start+j].relist_idx, key);
        }
    
        gettimeofday(&end1, NULL);

        cpu_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
       //test
        
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //    key_t key = host_keys[start + j];
        //    cout<<key<<": "<<val[j]<<endl;
        //}
       
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"GPU PPI V4 [sort key first,2threads,doublebuffer, whole]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;

    cout<<"CPU time                 "<<cpu_time<<endl;
    cout<<"GPU time(one stream)     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;
    
    
    CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys_after));
    CUDA_ERROR_HANDLER(cudaFree(d_temp_storage));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

/*}}}*/

}

#endif
