#include<cuda_runtime.h>
#include<iostream>
#include<sys/time.h>
#include <assert.h>

#include <string>
#include"ppi-bpt.h"
#include"cuda_utils.h"
#include"mempool.h"
#include<sched.h>
#include<pthread.h>
#include<unistd.h>

#include <fstream>
#include <omp.h>
#include "cub/cub.cuh"

#define M 1000000
#define Keys_Count 100*M

#define  GPU_SWP_SIZE2 16 


#define PPI_Block_Dim_2thread 128
#define PPI_Block_Per_SM 64

#define PPI_Thread_Per_Query_2thread 2
#define  Sort_Per_2Thread (GPU_SWP_SIZE2 / PPI_Thread_Per_Query_2thread)

#define CPU_THREAD 54 

#define LOAD_BALANCE_LEVEL 1 //CPU maybe compute last N level inner_node
#define LOAD_BALANCE_PECENTAGE 0 //unit %; Range 0~1 ,0 means CPU compute 0% inner_node,100% means CPU compute compute all inner_node
typedef struct{
    int blfnode = 0; //默认rootIdx

    //BLeaf_node *blfnode;
    //char relist_idx = 111;
    char relist_idx;
}GPU_Result;


using namespace std;
namespace{
    volatile int stream_flag[2] = {-1,-1};/*{{{*/
    int Thread_Per_Block = PPI_Block_Dim_2thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (Block_Per_Grid * Thread_Per_Block) / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;

    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;

    key_t *host_keys;
    int *host_pos; 
    GPU_Result  *h_gresult;


    key_t *d_keys;
    key_t *d_keys_after;
    int *d_pos;
    int *d_pos_after;
    GPU_Result *d_gresult;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    int rootIdx; 

    int kernel_height;
    int inner_node_size_wo_last_inner_node;
    int total;

    Inner_node *d_innode;
    int *d_prefix;
    cudaStream_t stream[2];


    void CUDART_CB CallBack(cudaStream_t stream,cudaError_t status,void *data){
        stream_flag[(size_t)data%2] = (size_t)data;
    }

    float time_gpu = 0;

    cudaEvent_t g_start,g_stop;

    volatile int key_status = -1;

    BPlusTree *bptree;
    vector<double> cpu_thread_total_time(CPU_THREAD,0);
    vector<double> cpu_thread_compute_time(CPU_THREAD,0);
    
    int bucket_balance_keys = ((int)((Para_Search_Bucket * LOAD_BALANCE_PECENTAGE+1024-1)/ 1024)) * 1024;
/*}}}*/
}

//---------------------------------------------------------------


__global__ void ppi_bpt_V7_search_kernel_2threads_new_tree(Inner_node *d_innode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count,int d_bucket_balance_keys){
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
    
    // for GPU CPU balance;
    bool isBalance = key_base *blockIdx.x < d_bucket_balance_keys;
    if(isBalance)
        tree_height = tree_height - LOAD_BALANCE_LEVEL;

    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE2;j++){
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
                    //result.blfnode = (BLeaf_node *)node->child[0];
                    if(isBalance){
                        result.blfnode = __ldg(&d_prefix[pos])+(begin+8) ;
                        result.relist_idx = 111;

                    }
                    else{
                        result.blfnode = pos - inner_node_size_wo_last_inner_node;//for leaf
                        result.relist_idx = begin+8;
                    } 
                }else
                    start_idx[cur_r] = __ldg(&d_prefix[pos]) + (begin+8);
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
                    if(isBalance){
                        result.blfnode = __ldg(&d_prefix[pos])+idx+nexthalf[r]*2 ;
                        result.relist_idx = 111;
                   }else{ 
                        //result.blfnode = pos - inner_node_size_wo_last_inner_node;//useless
                        result.relist_idx = idx + nexthalf[r]*2;
                   }    

                }else 
                    start_idx[cur_r] = __ldg(&d_prefix[pos]) + (idx + nexthalf[r]*2);
            }
            inner_index_result[r] = 0;
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            __syncthreads();
        
        }
    }/*}}}*/
}




/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V7
*           double buffer
*           using 2 thread
*           batch
*           key sort first, pos back
*           whole
*           new tree 
*            CPU multi-thread
*           load-balance ;GPU up,cpu down;
*
*-----------------------------------------------------------------------------*/

void* launch_kernel_thread_new_tree_measure_v7(void *args){
/*{{{*/
    stick_this_thread_to_core(1);
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

        ppi_bpt_V7_search_kernel_2threads_new_tree<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode,d_prefix, inner_node_size_wo_last_inner_node, rootIdx, d_keys_after+stride, kernel_height , d_gresult+stride, Para_Search_Bucket, bucket_balance_keys);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);


    }
    
    cudaEventRecord(g_stop);
   
       return NULL;/*}}}*/
}

void *cpu_search_v7(void *args){
/*{{{*/
    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time = 0;
    double compute_time = 0;
    int threadID = (long long)args; 
    stick_this_thread_to_core(threadID+2);

    const int keys = Para_Search_Bucket/ CPU_THREAD;// every thread process N keys for once;
    const int count = (Para_Search_Bucket+keys-1) / keys;//how many batch in bucket
    const int total_keys = (count+CPU_THREAD-1) / CPU_THREAD * keys;// keys count  for one thread
    value_t val[total_keys];
    
    void *nodes[total_keys];
    int relist_idxs[total_keys];
    
    gettimeofday(&start, NULL); 
    
    
    for (int i=0;i<total;i++) {
        int idx = i%2;
        while(stream_flag[idx]<i);
        int start = i * Para_Search_Bucket;

        int process_idx = start+threadID*keys;
        int j = 0;
        int balance_thresh = bucket_balance_keys+start;
        gettimeofday(&start1, NULL); 
        while(process_idx<start + Para_Search_Bucket){
            int k = process_idx;
            // CPU process inner_node
            int tmpj = j;
            for(;k < process_idx+keys && k < start+Para_Search_Bucket;k++){ 
                key_t key = host_keys[start + host_pos[k]];
                if(k < balance_thresh){ 
                    
                    nodes[j] = (void *)(bptree->getInnerNodeByIdx(h_gresult[k].blfnode));
                    relist_idxs[j] = 0;
                    for(int i=1;i<=LOAD_BALANCE_LEVEL;i++){
                         relist_idxs[j] = ((Inner_node *)nodes[j])->getChildIdx_avx2(NULL,key);
                        nodes[j] = ((Inner_node *)nodes[j])->child[relist_idxs[j]];
                        //__builtin_prefetch(nodes[j],0,3);
                    
                    }
                    j++;   
                }else{
                    BLeaf_node *blfnode = bptree->getLeafByIdx(h_gresult[k].blfnode);
                    val[j++] = blfnode->findKey(h_gresult[k].relist_idx,key);
                } 
            }
            //CPU process bleaf_node
            
            k = process_idx;
            for(;k < process_idx+keys && k < start+Para_Search_Bucket;k++){
                if(k >= balance_thresh) break;
                key_t key = host_keys[start + host_pos[k]];
                val[tmpj]=((BLeaf_node *)nodes[tmpj])->findKey(relist_idxs[tmpj],key);
                tmpj++;
            }

            process_idx +=  CPU_THREAD * keys;
        }
        gettimeofday(&end1, NULL); 
        compute_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;  
               //test
       // process_idx = start+threadID*keys;
       // j = 0;
       // while(process_idx<Para_Search_Bucket){
       //     int k = process_idx;
       //     for(;k<process_idx+keys && k<Para_Search_Bucket;k++){ 
       //         key_t key = host_keys[start + host_pos[k]];
       //         if(val[j]!=key)
       //             cout<<key<<" "<<val[j]<<endl;
       //         j++;
       //     }
       //     process_idx +=  CPU_THREAD * keys;
       // }
 
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;  
    cpu_thread_total_time[threadID] = total_time;
    cpu_thread_compute_time[threadID] = compute_time;
    return 0;/*}}}*/
}

void PPI_BPT_Search_GPU_V7_2thread_new_tree_CPUMultiThread(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    stick_this_thread_to_core(0);
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    
    
    bptree = &tree;
    prepareGPU_v2(tree, d_innode, d_prefix);

    rootIdx = tree.getRootIdx();
    inner_node_size_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();


    kernel_height = tree.getHeight();
    assert(rootIdx != -1);
   
    //for balance 
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
    
    cout<<bucket_balance_keys<<endl;
    cout<<Para_Search_Bucket<<endl;
    cout<<nums<<endl;


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
    if((pthread_create(&tid,NULL,launch_kernel_thread_new_tree_measure_v7,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
    
   vector<pthread_t> tid_arr;
   for(int i=0;i< CPU_THREAD;i++){
       if((pthread_create(&tid,NULL,cpu_search_v7,(void *)i))!=0){
           cout<<"can't create thread\n"<<endl;
       }else{
           tid_arr.push_back(tid);
       }
   }

   
   
   gettimeofday(&start, NULL); 
   for(int i= 0;i<CPU_THREAD;i++){
       pthread_join(tid_arr[i],NULL);
   }
   gettimeofday(&end, NULL);
   total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;  

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"GPU PPI V7 [sort key first,2threads,doublebuffer, whole, new_tree,load balance()]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU time(one stream)     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;
    //compute time  
    double tmp = 0;
    for(auto t:cpu_thread_compute_time){
        //cout<<"compute_time:"<<t<<endl;
        tmp+=t;
    }
    cout<<"average thread compute time:"<<tmp/ CPU_THREAD<<endl;
    //for(int i=0;i<CPU_THREAD;i++){
    //    cout<<i<<":"<<cpu_thread_compute_time[i]<<endl;
    //}
    tmp = 0;
    for(auto t:cpu_thread_total_time){
        tmp+=t;
    }
    cout<<"average thread total_time "<<tmp/ CPU_THREAD<<endl;
   
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



