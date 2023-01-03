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

#define  GPU_SWP_SIZE (1*Thread_Per_Query)  //old 8


#define Block_Dim 128
#define Grid_Dim (PPI_Block_Per_SM * SM)

//#define Narrow_Rate 4 // 1/Narrow_Rate fanout
#define Thread_Per_Query (DEFAULT_ORDER/Narrow_Rate)

#define Sort_Per_Thread (GPU_SWP_SIZE / Thread_Per_Query)


typedef struct{
    r_ptr_t record_ptr;
}GPU_Result;


using namespace std;
namespace prefix_sort{
/*{{{*/
    

    int Para_Search_Bucket = (Grid_Dim * Block_Dim) / Thread_Per_Query * GPU_SWP_SIZE;//851968

    dim3 block_dim(Block_Dim);
    dim3 grid_dim(Grid_Dim);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;

    key_t **host_keys;
    int **host_pos; 
    GPU_Result  **h_gresult;
    
    key_t **d_keys;
    key_t **d_keys_after;
    int **d_pos;
    int **d_pos_after;
    GPU_Result **d_gresult;
    
    
    //void *d_temp_storage = NULL;
    void **d_temp_storage;
    size_t temp_storage_bytes = 0;

    int rootIdx; 
    int internal_node_num;
    int kernel_height;
    int total;
    int total_per_gpu;

    kArr_t **d_node;
    int **d_prefix;
    cArr_t **d_leaf_record;
    
    
    cudaStream_t **stream;

    float time_gpu = 0;

    //cudaEvent_t g_start,g_stop;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;

    int ngpus;


/*}}}*/
} ;
using namespace prefix_sort;
//---------------------------------------------------------------
//  leaf: height=0
//  *d_leaf_record only contains leaf's child record, so it starts from internal_node_num of pointer section
__global__ void multi_prefix_sort_kernel(kArr_t *d_node, int* d_prefix, cArr_t *d_leaf_record, int root_idx, int internal_node_num, key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
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
            
            int tmpMask = 0xffffffff;
            for (int k=0; k<Narrow_Rate; k++) {
                
                selfFlag = 0;
                flag[r][search_idx+1] = 0;
                
                key_t idx_key = node->key[search_idx + k*Thread_Per_Query];

                if (target<idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = k;
                }
                __syncwarp(tmpMask);
                tmpMask = __ballot_sync(tmpMask, nexthalf[r]==-1);

                if (nexthalf[r]!=-1) break; 



            }
            __syncwarp();
           /// __syncthreads();
             
            if (selfFlag == 1 && flag[r][search_idx] ==0) {
                start_idx[cur_r] = __ldg(&d_prefix[pos]) + nexthalf[r]*Thread_Per_Query + search_idx;
            }
            __syncthreads();

        }
    }
    //leaf
    for(int j=0;j<GPU_SWP_SIZE;j++){
        nexthalf[r] = -1;
        __syncthreads();
        
        int cur_key = key_idx + stride * j;
        if (cur_key>=key_count) continue;
        
        int cur_r = r + stride*j;
        
        target = d_keys[cur_key];
        GPU_Result &result = d_gresult[cur_key];
        
        int pos = start_idx[cur_r];
        kArr_t *node = d_node + pos;
        
        key_t idx_key;
        int tmpMask = 0xffffffff;
        for (int k=0; k<Narrow_Rate; k++) {
            
            selfFlag = 0;
            flag[r][search_idx+1] = 0;
            idx_key = node->key[search_idx + k*Thread_Per_Query];

            if (target<=idx_key){
                flag[r][search_idx+1] = 1;
                selfFlag = 1;
                nexthalf[r] = k;
            
            }
            __syncwarp(tmpMask);
            tmpMask = __ballot_sync(tmpMask, nexthalf[r]==-1);


            if (nexthalf[r]!=-1) break; 



        }
        
         
        __syncwarp();
        //__syncthreads();
        if (selfFlag == 1 && flag[r][search_idx] ==0) {
            //printf("%lld, %lld\n", idx_key, target);
            if (idx_key == target){
                result.record_ptr = d_leaf_record[pos - internal_node_num].record[nexthalf[r]*Thread_Per_Query+search_idx];
            }
            else {    
                result.record_ptr = -1;
            }
        }
        __syncthreads();


    }
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /*}}}*/
}




/*--------------------------------------------------------------------------
*
*       multi-prefix-sort 
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
            const int *tmp_dpos =d_pos[iii]+stride;
        
        
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii] + stride, host_keys[iii] + i * Para_Search_Bucket, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][idx]));
        
#ifdef TREE_32
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after[iii]+stride, tmp_dpos, d_pos_after[iii]+stride, Para_Search_Bucket,16,32,stream[iii][idx]);
#else
        
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, tmp_dkeys, d_keys_after[iii]+stride, tmp_dpos, d_pos_after[iii]+stride, Para_Search_Bucket,48,64,stream[iii][idx]);
#endif        
            multi_prefix_sort_kernel<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_node[iii],d_prefix[iii],d_leaf_record[iii], rootIdx, internal_node_num, d_keys_after[iii]+stride, kernel_height , d_gresult[iii]+stride, Para_Search_Bucket);
            
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+i*Para_Search_Bucket, d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos[iii]+i*Para_Search_Bucket, d_pos_after[iii]+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            
        }
    }
        
        
        
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        cudaEventRecord(g_stop[iii]);
    }
    return NULL;/*}}}*/
}

void multi_prefix_sort(CFB &tree, ifstream &search_file){
/*{{{*/
    cout<<DEFAULT_ORDER<<" "<<Narrow_Rate<<" "<<GPU_SWP_SIZE<<endl;

    //cudaGetDeviceCount(&ngpus);
    ngpus=  GPU_NUM;
    d_node = (kArr_t **)malloc(ngpus * sizeof(kArr_t*));
    d_prefix = (int **)malloc(ngpus * sizeof(int*));
    d_leaf_record = (cArr_t **)malloc(ngpus * sizeof(cArr_t*));

    for (int i=0; i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        prepareGPU(tree, d_node[i], d_prefix[i], d_leaf_record[i]);
    }

    rootIdx = tree.getRootIdx();
    internal_node_num = tree.internal_node_num;
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



    cout<<"Bucket size "<<Para_Search_Bucket<<endl;
    //cout<<nums<<endl;

    h_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    host_pos = (int **)malloc(ngpus * sizeof(int*));
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
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
        
        cudaSetDevice(iii+GPU_START);
        
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


    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaMemcpy(d_pos[i], host_pos[i], batch_pos_size*2, cudaMemcpyHostToDevice));
    }
    
    
    d_temp_storage = (void **)malloc(ngpus * sizeof(void*));
    d_temp_storage[0] = NULL;
    cudaSetDevice(0+GPU_START); 
    cub::DeviceRadixSort::SortPairs(d_temp_storage[0], temp_storage_bytes, d_keys[0], d_keys_after[0], d_pos[0],d_pos_after[0],Para_Search_Bucket,16,32);
    
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_temp_storage[i]), temp_storage_bytes*2));
    }
    


    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
   
    //----------------------------------------------- 
    cout<<"multigpu prefix sort 1/"<<Narrow_Rate<<" thread "<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
   
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
                key_t key = host_keys[iii][base + host_pos[iii][idx]];
                int record_ptr=h_gresult[iii][idx].record_ptr;
                val_t v;
                if (record_ptr == -1) v=-1;
                else v = tree.record_section->elementAtIdx(h_gresult[iii][idx].record_ptr)->value;
                cout<<"key: "<<key<<" value "<<v<<endl;
            }
        }
    
    
    }


    #endif
    
    
    
    for (int i=0;i<ngpus;i++) {
        
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_pos[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_node[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_prefix[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_leaf_record[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_after[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_temp_storage[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
    }

/*}}}*/

}



