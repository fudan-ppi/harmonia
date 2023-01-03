#include<cuda_runtime.h>
#include<iostream>
#include<sys/time.h>
#include <assert.h>

#include <string>
#include"../src/cuda_utils.h"
#include"mempool.h"
//#include<sched.h>
//#include<pthread.h>
//#include<unistd.h>

#include <fstream>
//#include <omp.h>
#include "cub/cub.cuh"

#include "helper.h"
#include "../src/conf.h"
#include "gpu-cfb.h"

#define M 1000000
#define Keys_Count 100*M

#define  GPU_SWP_SIZE (8*Thread_Per_Query) 


#define Block_Dim 128
#define Grid_Dim (PPI_Block_Per_SM * SM)

//#define Narrow_Rate 4 // 1/Narrow_Rate fanout
#define Thread_Per_Query (DEFAULT_ORDER/Narrow_Rate)

#define Sort_Per_Thread (GPU_SWP_SIZE / Thread_Per_Query)


typedef struct{
    r_ptr_t record_ptr;
}GPU_Result;


using namespace std;
namespace half_noprefix_nosort_RB{
/*{{{*/
    

    int Para_Search_Bucket = (Grid_Dim * Block_Dim) / Thread_Per_Query * GPU_SWP_SIZE;//851968

    dim3 block_dim(Block_Dim);
    dim3 grid_dim(Grid_Dim);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;

    key_t **host_keys;
    GPU_Result  **h_gresult;
    
    key_t **d_keys;
    GPU_Result **d_gresult;
    

    int rootIdx; 
    int kernel_height;
    int total;
    int total_per_gpu;

    kArr_t **d_node;
    cArr_t **d_child;
    
    
    cudaStream_t **stream;

    float time_gpu = 0;

    //cudaEvent_t g_start,g_stop;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;

    int ngpus;


/*}}}*/
} ;
using namespace half_noprefix_nosort_RB;
//---------------------------------------------------------------
//  leaf: height=0

__global__ void half_multi_noprefix_nosort_RB_kernel(kArr_t *d_node, cArr_t *d_child, int root_idx, key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 

    int key_base = Block_Dim / Thread_Per_Query * GPU_SWP_SIZE;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ Thread_Per_Query;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = Block_Dim/ Thread_Per_Query;//blockDim.x/ Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE;
    __shared__ char flag[row][Thread_Per_Query+1];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = Block_Dim / Thread_Per_Query;
    
    for (int k = 0; k<Sort_Per_Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }

    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target;
    
    // for GPU CPU balance; without leaf
    for(int i=0;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE;j++){
            nexthalf[r] = -1;
            __syncthreads();
            
            int cur_key = key_idx + stride * j;
            if (cur_key>=key_count) continue;
            
            int cur_r = r + stride*j;
            
            target = d_keys[cur_key];
            
            int pos = start_idx[cur_r];
            kArr_t *node = d_node + pos;
            

            for (int k=0; k<Narrow_Rate; k++) {
                
                selfFlag = 0;
                flag[r][search_idx+1] = 0;
                
                key_t idx_key = node->key[search_idx + k*Thread_Per_Query];

                if (target<idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = k;
                }
                __syncthreads();

                if (nexthalf[r]!=-1) break; 



            }
            
            __syncthreads();
             
            if (selfFlag == 1 && flag[r][search_idx] ==0) {
                start_idx[cur_r] = d_child[pos].child[nexthalf[r]*Thread_Per_Query + search_idx];
            }
            __syncthreads();

        }
    }
        
    
    
    /*}}}*/
}




/*--------------------------------------------------------------------------
*
*       half-multi-noprefix-nosort_RB 
*
*-----------------------------------------------------------------------------*/

static void* launch_kernel_thread(void *args){
/*{{{*/
    g_start = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
    g_stop = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
    
    
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        cudaEventCreate(&(g_start[iii]));
        cudaEventCreate(&(g_stop[iii]));
        cudaEventRecord(g_start[iii]);
    }
 
    
    
    for(int i=0;i<total_per_gpu;i++){
        for (int iii=0;iii<ngpus;iii++) {
            cudaSetDevice(iii+GPU_START);
            
            int idx = i%2;
            int stride = idx*Para_Search_Bucket;
            const key_t *tmp_dkeys =d_keys[iii]+stride;
        
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii] + stride, host_keys[iii] + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][idx]));
        
        
        
            half_multi_noprefix_nosort_RB_kernel<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_node[iii],d_child[iii], rootIdx, d_keys[iii]+stride, kernel_height , d_gresult[iii]+stride, Para_Search_Bucket);
            
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+i*Para_Search_Bucket, d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            
        }
    }
        
        
        
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        cudaEventRecord(g_stop[iii]);
    }
    return NULL;/*}}}*/
}

void half_multi_noprefix_nosort_RB(RB &tree, ifstream &search_file){
/*{{{*/
    cout<<DEFAULT_ORDER<<" "<<Narrow_Rate<<" "<<GPU_SWP_SIZE<<endl;

    //cudaGetDeviceCount(&ngpus);
    ngpus=  GPU_NUM;
    d_node = (kArr_t **)malloc(ngpus * sizeof(kArr_t*));
    d_child = (cArr_t **)malloc(ngpus * sizeof(cArr_t*));

    for (int i=0; i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        prepareGPU_noprefix_RB(tree, d_node[i], d_child[i]);
    }

    rootIdx = tree.getRootIdx();
    kernel_height = tree.getHeight();
    
    
    assert(rootIdx != -1);
   
    //for balance 
    //host_malloc
    host_keys = (key_t **)malloc(ngpus * sizeof(key_t *));
    int nKeys_Count = Keys_Count / ngpus;
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
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
  // 这里使用nKeys_Count(而不是nums/ngpus)是为了使得在nums不是Keys_Count大小时依然在结果上正确。因为任务不是按照保证每个GPU做同量分配的，而是保证前面的gpu可以占满



    //cout<<Para_Search_Bucket<<endl;
    //cout<<nums<<endl;

    h_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaMallocHost(&(h_gresult[i]),sizeof(GPU_Result)*nKeys_Count));
    }



    stream = (cudaStream_t **)malloc(ngpus * sizeof(cudaStream_t*));
    d_keys = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    for (int iii=0; iii<ngpus; iii++) {
        
        cudaSetDevice(iii+GPU_START);
        
        stream[iii] = (cudaStream_t *)malloc(2 * sizeof(cudaStream_t));
        for(int i=0;i<2;i++) cudaStreamCreate(&(stream[iii][i]));
    
    
    
        //gpu_malloc
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys[iii]), batch_d_key_size*2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_gresult[iii]), batch_gresult_size *2));
    }

    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
   
    //----------------------------------------------- 
    cout<<"half multigpu noprefix nosort_RB 1/"<<Narrow_Rate<<" thread"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
   
    pthread_join(tid, NULL); 

    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        cudaEventSynchronize(g_stop[i]);
          
        cudaEventElapsedTime(&time_gpu,g_start[i],g_stop[i]);
        cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    }
    
    //get back
    #ifdef ENABLE_TEST
    
    for (int iii=0;iii<ngpus;iii++) {
        for (int i=0;i<total_per_gpu;i++) {
            int base = i*Para_Search_Bucket;
            for (int j=0;j<Para_Search_Bucket;j++){
                int idx = base + j;
                key_t key = host_keys[iii][idx];
                int record_ptr=h_gresult[iii][idx].record_ptr;
                val_t v;
                if (record_ptr == -1) v=-1;
                else v = tree.record_section->elementAtIdx(record_ptr)->value;
                cout<<"key: "<<key<<" value "<<v<<endl;
            }
        }
    
    
    }


    #endif
    
    
    
    for (int i=0;i<ngpus;i++) {
        
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_node[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_child[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
    }

/*}}}*/

}



