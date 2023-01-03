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
typedef struct{
    int blfnode;

    //BLeaf_node *blfnode;
    char relist_idx;
}GPU_Result;


using namespace std;


__global__ void ppi_bpt_V10_search_kernel_2threads_new_tree(Inner_node *d_innode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
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
                    result.blfnode = pos - inner_node_size_wo_last_inner_node;
                    result.relist_idx = begin+8;
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

                    //result.blfnode = pos - inner_node_size_wo_last_inner_node;
                    result.relist_idx = idx + nexthalf[r]*2;
                    //return;
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



namespace{
/*{{{*/

    volatile int** stream_flag;
    //volatile int stream_flag[2] = {-1,-1};    
    int Thread_Per_Block = PPI_Block_Dim_2thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (Block_Per_Grid * Thread_Per_Block) / PPI_Thread_Per_Query_2thread * GPU_SWP_SIZE2;

    dim3 block_dim(Thread_Per_Block);
    dim3 grid_dim(Block_Per_Grid);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;

    //key_t *host_keys;
    //int *host_pos; 
    //GPU_Result  *h_gresult;
    key_t **host_keys;
    int **host_pos; 
    GPU_Result  **h_gresult;

    //key_t *d_keys;
    //key_t *d_keys_after;
    //int *d_pos;
    //int *d_pos_after;
    //GPU_Result *d_gresult;

    key_t **d_keys;
    key_t **d_keys_after;
    int **d_pos;
    int **d_pos_after;
    GPU_Result **d_gresult;

    //void *d_temp_storage = NULL;
    void **d_temp_storage;
    size_t temp_storage_bytes = 0;

    int rootIdx; 

    int kernel_height;
    int inner_node_size_wo_last_inner_node;
    int total;
    int total_per_gpu;

    //Inner_node *d_innode;
    //int *d_prefix;
    Inner_node **d_innode;
    int **d_prefix;
    
    
    //cudaStream_t *stream[2];
    cudaStream_t **stream;


    void CUDART_CB CallBack(cudaStream_t stream,cudaError_t status,void *data){
        stream_flag[(size_t)data/total_per_gpu][(size_t)data%total_per_gpu%2] = (size_t)data;
    }

    float time_gpu = 0;

    //cudaEvent_t g_start,g_stop;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;


    BPlusTree *bptree;
    vector<double> cpu_thread_total_time(CPU_THREAD,0);
    vector<double> cpu_thread_compute_time(CPU_THREAD,0);

    int ngpus;
/*}}}*/
}


static void* launch_kernel_thread_new_tree_measure(void *args){
/*{{{*/

    stick_this_thread_to_core(1);
    
    g_start = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
    g_stop = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
       
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii);
        cudaEventCreate(&(g_start[iii]));
        cudaEventCreate(&(g_stop[iii]));
        cudaEventRecord(g_start[iii]);
    }
    for(int i=0;i<total_per_gpu;i++){
        for (int iii=0;iii<ngpus;iii++) {
            cudaSetDevice(iii);
            int idx = i%2;
            int stride = idx*Para_Search_Bucket;
            const key_t *tmp_dkeys =d_keys[iii]+stride;
            const int *tmp_dpos =d_pos[iii]+stride;
        

            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii] + stride, host_keys[iii] + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][idx]));

#ifdef TREE_32
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after[iii]+stride, tmp_dpos, d_pos_after[iii]+stride, Para_Search_Bucket,16,32,stream[iii][idx]);

#else 
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after[iii]+stride, tmp_dpos, d_pos_after[iii]+stride, Para_Search_Bucket,48,64,stream[iii][idx]);
#endif  

            ppi_bpt_V10_search_kernel_2threads_new_tree<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_prefix[iii], inner_node_size_wo_last_inner_node, rootIdx, d_keys_after[iii]+stride, kernel_height , d_gresult[iii]+stride, Para_Search_Bucket);

            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+i*Para_Search_Bucket, d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos[iii]+i*Para_Search_Bucket, d_pos_after[iii]+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][idx]));

            cudaStreamAddCallback(stream[iii][idx],CallBack,(void *)(i+iii*total_per_gpu),0);


        }
    
   
    }
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii);
        cudaEventRecord(g_stop[iii]);
    }
    
    
    return NULL;/*}}}*/
}



static void *cpu_search(void *args){
/*{{{*/

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time = 0;
    double compute_time = 0;
    int threadID = (long long)args; 
 //   if (threadID==0)
    stick_this_thread_to_core(threadID+2);

    //const int keys = 16;
    const int keys = Para_Search_Bucket/ CPU_THREAD;// every thread process 16 key for once;
    const int count = (Para_Search_Bucket+keys-1) / keys;//how many batch in bucket
    const int total_keys = (count+CPU_THREAD-1) / CPU_THREAD * keys;// keys count  for one thread
    value_t val[total_keys];

    gettimeofday(&start, NULL); 
   
    
    for (int i=0;i<total_per_gpu;i++) {

        int idx = i%2;
        int start = i * Para_Search_Bucket;
        
        for (int iii=0;iii<ngpus;iii++) {
            while(stream_flag[iii][idx]<i+iii*total_per_gpu);
        
            int process_idx = start+threadID*keys;
        
            int j = 0;
        
            gettimeofday(&start1, NULL); 
            while(process_idx<start + Para_Search_Bucket){
                int k = process_idx;
                for(;k < process_idx+keys && k < start+Para_Search_Bucket;k++){ 
                    key_t key = host_keys[iii][start + host_pos[iii][k]];
                    BLeaf_node *blfnode = bptree->getLeafByIdx(h_gresult[iii][k].blfnode);
                    val[j++] = blfnode->findKey(h_gresult[iii][k].relist_idx, key);
                }
                process_idx +=  CPU_THREAD * keys;
            }
            
            gettimeofday(&end1, NULL); 
        
            compute_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;  
        
    //test
    //       process_idx = start+threadID*keys;
    //       j = 0;
    //       while(process_idx<start + Para_Search_Bucket){
    //           int k = process_idx;
    //           for(;k<process_idx+keys && k<start + Para_Search_Bucket;k++){ 
    //               key_t key = host_keys[iii][start + host_pos[iii][k]];
    //                cout<<key<<" "<<val[j++]<<endl;
    //           }
    //           process_idx +=  CPU_THREAD * keys;
    //       }

        
        }

   
    }
 //   if(threadID == 0)
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;  
    cpu_thread_total_time[threadID] = total_time;
    cpu_thread_compute_time[threadID] = compute_time;


    return 0;/*}}}*/
}


/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V10
*           double buffer
*           using 2 thread
*           batch
*           key sort first,pos back
*           whole
*           new tree
*           CPU multi-thread
*           multigpu
*
*-----------------------------------------------------------------------------*/


void PPI_BPT_Search_GPU_V10_multigpu_basedv5(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    stick_this_thread_to_core(0);
    
    cudaGetDeviceCount(&ngpus);


    bptree = &tree;

    d_innode = (Inner_node **)malloc(ngpus * sizeof(Inner_node*));
    d_prefix = (int **)malloc(ngpus * sizeof(int*));
    
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        prepareGPU_v2(tree, d_innode[i], d_prefix[i]);
    }

    rootIdx = tree.getRootIdx();
    inner_node_size_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();
    kernel_height = tree.getHeight();
    
    
    assert(rootIdx != -1);
    
    //host_malloc
    host_keys = (key_t **)malloc(ngpus * sizeof(key_t *));
    int nKeys_Count = Keys_Count / ngpus;
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_keys[i]),sizeof(key_t)*nKeys_Count));
    }

    //init key 
    int nums = 0;
    string s;
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums/nKeys_Count][nums%nKeys_Count] = key;
        nums++;
    } 
    total = nums / Para_Search_Bucket;
    total_per_gpu = nKeys_Count / Para_Search_Bucket;

    h_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    host_pos = (int **)malloc(ngpus * sizeof(int*));
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        CUDA_ERROR_HANDLER(cudaMallocHost(&(h_gresult[i]),sizeof(GPU_Result)*nKeys_Count));
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_pos[i]),sizeof(int)*nKeys_Count));
    }

    
    stream = (cudaStream_t **)malloc(ngpus * sizeof(cudaStream_t*));
    d_keys = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_keys_after = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_pos = (int **)malloc(ngpus * sizeof(int*));
    d_pos_after = (int **)malloc(ngpus * sizeof(int *));
    d_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    for (int iii=0; iii<ngpus; iii++) {
        
        cudaSetDevice(iii);
        
        //init key pos
        for (int i=0;i<Para_Search_Bucket;i++) {
            host_pos[iii][i] = i;
            host_pos[iii][i+Para_Search_Bucket] = i;
        }
        
        stream[iii] = (cudaStream_t *)malloc(2 * sizeof(cudaStream_t));
        for(int i=0;i<2;i++) cudaStreamCreate(&(stream[iii][i]));
    
    
    
        //gpu_malloc
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys[iii]), batch_d_key_size*2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys_after[iii]), batch_d_key_size*2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_pos[iii]), batch_pos_size*2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_pos_after[iii]), batch_pos_size*2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_gresult[iii]), batch_gresult_size *2));
    }
    
        

    struct timeval start;
    struct timeval end;
    double total_time=0;
    double cpu_time = 0;
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        CUDA_ERROR_HANDLER(cudaMemcpy(d_pos[i], host_pos[i], batch_pos_size*2, cudaMemcpyHostToDevice));
    }

    d_temp_storage = (void **)malloc(ngpus * sizeof(void*));
    
    
    d_temp_storage[0] = NULL;
    cudaSetDevice(0); 
    cub::DeviceRadixSort::SortPairs(d_temp_storage[0], temp_storage_bytes, d_keys[0], d_keys_after[0], d_pos[0],d_pos_after[0],Para_Search_Bucket,16,32);


    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_temp_storage[i]), temp_storage_bytes*2));
    }

 
    stream_flag = (volatile int **)malloc(ngpus * sizeof(int*)); 
    for (int i=0;i<ngpus;i++) {
        stream_flag[i] = (volatile int*)malloc(2*sizeof(int));
        stream_flag[i][0] = -1;
        stream_flag[i][1] = -1;
    }


    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread_new_tree_measure,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
    
    vector<pthread_t> tid_arr;
    for(int i=0;i< CPU_THREAD;i++){
        if((pthread_create(&tid,NULL,cpu_search,(void *)i))!=0){
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


    cout<<"GPU PPI V10 [sort key first,2threads,doublebuffer, whole, new_tree, multi-gpu]"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
    
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i);
        cudaEventSynchronize(g_stop[i]);
          
        cudaEventElapsedTime(&time_gpu,g_start[i],g_stop[i]);
        cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    }
    
    cout<<"total_time:              "<<total_time<<endl;
    //compute time  
    double tmp = 0;
    for(auto t:cpu_thread_compute_time){
        tmp+=t;
    }
    cout<<"average thread compute time: "<<tmp/ CPU_THREAD<<endl;
    tmp = 0;
    for(auto t:cpu_thread_total_time){
        tmp+=t;
    }
    cout<<"average thread total_time: "<<tmp/ CPU_THREAD<<endl;
  
    for (int i=0;i<ngpus;i++) {
        
        cudaSetDevice(i);
        CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_pos[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_innode[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_after[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_temp_storage[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
    }

/*}}}*/

}



