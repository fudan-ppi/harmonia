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

//#include "papi.h"
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

#ifdef PAPI
//-----------------------papi------------------------------------

int EventSet = PAPI_NULL;

long long values1[10];
long long values2[10];
inline void papi_init() {
/*{{{*/
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        exit(1);
    if (PAPI_thread_init(pthread_self) != PAPI_OK)
        exit(1);


    /* Create an EventSet */


    int retval = PAPI_create_eventset (&EventSet);

    assert(retval==PAPI_OK);

    /* Add an event*/
/*
    retval = PAPI_add_event(EventSet,PAPI_L1_DCM);  //PAPI_L1_DCM  0x80000000  Yes   No   Level 1 data cache misses
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L1_TCM);  //PAPI_L1_TCM  0x80000006  Yes   Yes  Level 1 cache misses
    assert(retval==PAPI_OK);
    
    retval = PAPI_add_event(EventSet,PAPI_L1_LDM);  //PAPI_L1_LDM  0x80000017  Yes   No   Level 1 load misses
    assert(retval==PAPI_OK);

*/
   /* 
    retval = PAPI_add_event(EventSet,PAPI_L2_DCM);  //PAPI_L2_DCM  0x80000002  Yes   Yes  Level 2 data cache misses
    assert(retval==PAPI_OK);

    
    retval = PAPI_add_event(EventSet,PAPI_L2_TCM);  //PAPI_L2_TCM  0x80000007  Yes   No   Level 2 cache misses
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L2_LDM);  //PAPI_L2_LDM  0x80000019  Yes   No   Level 2 load misses
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L2_DCR);  //PAPI_L2_DCR  0x80000044  Yes   No   Level 2 data cache reads
    assert(retval==PAPI_OK);
    */
/*
    retval = PAPI_add_event(EventSet,PAPI_L2_DCW);  //PAPI_L2_DCW  0x80000047  Yes   No   Level 2 data cache writes
    assert(retval==PAPI_OK);
  */  

    
    retval = PAPI_add_event(EventSet,PAPI_L3_TCM);  //PAPI_L3_TCM  0x80000008  Yes   No   Level 3 cache misses
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L3_LDM);  //PAPI_L3_LDM  0x8000000e  Yes   No   Level 3 load misses
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L3_DCR);  //PAPI_L3_DCR  0x80000045  Yes   No   Level 3 data cache reads
    assert(retval==PAPI_OK);

    retval = PAPI_add_event(EventSet,PAPI_L3_DCW);  //PAPI_L3_DCW  0x80000048  Yes   No   Level 3 data cache writes
    assert(retval==PAPI_OK);
    

    
 //   retval = PAPI_add_event(EventSet,PAPI_STL_ICY); //PAPI_STL_ICY 0x80000025  Yes   No   Cycles with no instruction issue
 //   assert(retval==PAPI_OK);

 //   retval = PAPI_add_event(EventSet,PAPI_STL_CCY); //PAPI_STL_CCY 0x80000027  Yes   No   Cycles with no instructions completed
 //   assert(retval==PAPI_OK);

 //   retval = PAPI_add_event(EventSet,PAPI_TOT_INS); //PAPI_TOT_INS 0x80000032  Yes   No   Instructions completed
 //   assert(retval==PAPI_OK);

 //   retval = PAPI_add_event(EventSet,PAPI_LST_INS); //PAPI_LST_INS 0x8000003c  Yes   Yes  Load/store instructions completed
 //   assert(retval==PAPI_OK);
/*
    retval = PAPI_add_event(EventSet,PAPI_L2_DCM);
    assert(retval==PAPI_OK);
    retval = PAPI_add_event(EventSet,PAPI_L2_DCM);
    assert(retval==PAPI_OK);*/
    /* Start counting events */

    if (PAPI_start(EventSet) != PAPI_OK)

        retval = PAPI_start (EventSet);

    assert(retval==PAPI_OK);


    PAPI_read(EventSet, values1);

    assert(retval==PAPI_OK);
}

inline void papi_end() {
    /* Stop counting events */

    int retval = PAPI_stop (EventSet,values2);

    assert(retval==PAPI_OK);

    /* Clean up EventSet */

    if (PAPI_cleanup_eventset(EventSet) != PAPI_OK)

           exit(-1);

    /* Destroy the EventSet */

    if (PAPI_destroy_eventset(&EventSet) != PAPI_OK)

          exit(-1);

    /* Shutdown PAPI */

    PAPI_shutdown();
    

    for (int i=0;i<10;i++) {
        values2[i] -= values1[i];
        cout<<values2[i]<<endl;
    }/*}}}*/
}



#endif






//---------------------------------------------------------------


void stick_this_thread_to_core(int core_id){
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);/*{{{*/
   
   //core_id*=2;
    //core_id = core_id % num_cores;
    static const int arr[] = {0,28,2,30,4,32,6,34,8,36,10,38,12,40,14,42,16,44,18,46,20,48,22,50,24,52,26,54,1,29,3,31,5,33,7,35,9,37,11,39,13,41,15,43,17,45,19,47,21,49,23,51,25,53,27,55};
    //static const int arr[] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25,27};
    if(core_id<0 || core_id >= num_cores) {
        cout<<"core id wrong"<<endl;   
    }
    core_id = arr[core_id];
    cpu_set_t mask;
    cpu_set_t get;
    CPU_ZERO(&mask);
    CPU_SET(core_id,&mask);
    if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)<0){
        cout<<"set thread affinity error"<<endl;
    }
    //else{
    //    CPU_ZERO(&get);
    //    if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0) {
    //            cout<< "get thread affinity failed\n"<<endl;
    //    }
    //    for (int i = 0; i < num_cores; i++) {
    //        if (CPU_ISSET(i, &get)) {
    //            printf("this thread %d is running in processor %d\n", (int)pthread_self(), i); 
    //        }   
    //    }   
    //}
/*}}}*/
}

__global__ void ppi_bpt_V5_search_kernel_2threads_new_tree(Inner_node *d_innode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
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




/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V5
*           using 2 thread
*           batch
*           key  sort first, pos back
*           whole
*           new tree
*
*-----------------------------------------------------------------------------*/

void PPI_BPT_Search_GPU_V5_2thread_new_tree_serial(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    Inner_node *d_innode;
    int *d_prefix;

    prepareGPU_v2(tree, d_innode, d_prefix);
    int rootIdx = tree.getRootIdx();
    int inner_node_size_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();
    
    
    assert(rootIdx != -1);
    int Thread_Per_Block = PPI_Block_Dim_2thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (13 * PPI_Block_Per_SM * Thread_Per_Block) / PPI_Thread_Per_Query_2thread *GPU_SWP_SIZE2;
    
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
        
        ppi_bpt_V5_search_kernel_2threads_new_tree<<<grid_dim, block_dim>>>(d_innode, d_prefix, inner_node_size_wo_last_inner_node, rootIdx , d_keys_after, kernel_height, d_gresult, Para_Search_Bucket);

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
  //      #pragma omp parallel for
        for (int j=0; j<Para_Search_Bucket;j++) {
            key_t key = host_keys[host_pos[j]+ss];
            //BLeaf_node *blfnode = h_gresult[j].blfnode;
            BLeaf_node *blfnode = tree.getLeafByIdx(h_gresult[j].blfnode);
            val[j] = blfnode->findKey(h_gresult[j].relist_idx, key);
        }
    
        gettimeofday(&end, NULL);
        t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
        //test
        //for (int j=0;j<Para_Search_Bucket;j++) {
        //  key_t key = host_keys[host_pos[j]+ss];
        //
        //    cout<<key<<": "<<val[j]<<endl;
        //}
    }

    cout<<"GPU PPI V5 [sort key first, batch,2threads,whole, new_tree]"<<total * Para_Search_Bucket<<endl;
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

/*}}}*/
}



/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V5
*           double buffer
*           using 2 thread
*           batch
*           key sort first, pos back
*           whole
*           new tree 
*
*
*-----------------------------------------------------------------------------*/

void* launch_kernel_thread_new_tree_measure(void *args){
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

        ppi_bpt_V5_search_kernel_2threads_new_tree<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode,d_prefix, inner_node_size_wo_last_inner_node, rootIdx, d_keys_after+stride, kernel_height , d_gresult+stride, Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);


    }
    
    cudaEventRecord(g_stop);
   
       return NULL;/*}}}*/
}
void PPI_BPT_Search_GPU_V5_2thread_new_tree(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    prepareGPU_v2(tree, d_innode, d_prefix);

    rootIdx = tree.getRootIdx();
    inner_node_size_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();


    kernel_height = tree.getHeight();
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
    if((pthread_create(&tid,NULL,launch_kernel_thread_new_tree_measure,NULL))!=0){
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

            //BLeaf_node *blfnode = h_gresult[start+j].blfnode;
            BLeaf_node *blfnode = tree.getLeafByIdx(h_gresult[start+j].blfnode);
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
    cout<<"GPU PPI V5 [sort key first,2threads,doublebuffer, whole, new_tree]"<<total * Para_Search_Bucket<<endl;
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

/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V5
*           double buffer
*           using 2 thread
*           batch
*           key sort first,pos back
*           whole
*           new tree
*           CPU multi-thread
*
*
*-----------------------------------------------------------------------------*/

void *cpu_search(void *args){
/*{{{*/
    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time = 0;
    double compute_time = 0;
    int threadID = (long long)args; 
 //   if (threadID==0)
 //       papi_init();
    stick_this_thread_to_core(threadID+2);

    const int keys = Para_Search_Bucket/ CPU_THREAD;// every thread process 16 key for once;
    const int count = (Para_Search_Bucket+keys-1) / keys;//how many batch in bucket
    const int total_keys = (count+CPU_THREAD-1) / CPU_THREAD * keys;// keys count  for one thread
    value_t val[total_keys];

    gettimeofday(&start, NULL); 
    
    
    for (int i=0;i<total;i++) {
        int idx = i%2;
        while(stream_flag[idx]<i);
        int start = i * Para_Search_Bucket;

        int process_idx = start+threadID*keys;
        int j = 0;

        gettimeofday(&start1, NULL); 
        while(process_idx<start + Para_Search_Bucket){
            int k = process_idx;
            for(;k < process_idx+keys && k < start+Para_Search_Bucket;k++){ 
                key_t key = host_keys[start + host_pos[k]];
                BLeaf_node *blfnode = bptree->getLeafByIdx(h_gresult[k].blfnode);
                val[j++] = blfnode->findKey(h_gresult[k].relist_idx, key);
            }
            process_idx +=  CPU_THREAD * keys;
        }
        gettimeofday(&end1, NULL); 
        compute_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;  
               //test
//        process_idx = start+threadID*keys;
//        j = 0;
//        while(process_idx<start + Para_Search_Bucket){
//            int k = process_idx;
//            for(;k<process_idx+keys && k<start + Para_Search_Bucket;k++){ 
//                key_t key = host_keys[start + host_pos[k]];
//                 cout<<key<<" "<<val[j++]<<endl;
//            }
//            process_idx +=  CPU_THREAD * keys;
//        }
   
    }
 //   if(threadID == 0)
 //       papi_end();
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;  
    cpu_thread_total_time[threadID] = total_time;
    cpu_thread_compute_time[threadID] = compute_time;
    return 0;/*}}}*/
}
void PPI_BPT_Search_GPU_V5_2thread_new_tree_CPUMultiThread(BPlusTree &tree,ifstream &search_file){
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

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"GPU PPI V5 [sort key first,2threads,doublebuffer, whole, new_tree]"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU time(one stream)     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;
    //compute time  
    double tmp = 0;
    for(auto t:cpu_thread_compute_time){
        tmp+=t;
    }
    cout<<"average thread compute time:"<<tmp/ CPU_THREAD<<endl;
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



