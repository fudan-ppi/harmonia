#include<cuda_runtime.h>
#include<iostream>
#include<sys/time.h>
#include <assert.h>

#include <string>
#include"ppi-bpt.h"
#include"cuda_utils.h"
#include"mempool.h"
#include"buffer.h"

#include <fstream>
#include "cub/cub.cuh"
#include "conf.h"

#define M 1000000
#define Keys_Count 200*M

#define  GPU_SWP_SIZE 8 
//#define  GPU_SWP_SIZE (8*Thread_Per_Query) 


#define Block_Dim 128
#define Grid_Dim (PPI_Block_Per_SM *SM)


#define Thread_Per_Query 1
//#define Thread_Per_Query (DEFAULT_ORDER/Narrow_Rate) 
#define Sort_Per_Thread (GPU_SWP_SIZE / Thread_Per_Query)



typedef struct{
    int blfnode;
    //char relist_idx;    //0-63 
    //char r_idx;         //0-3 
    char idx;       //2 bit r_idx and 6 bit relist_idx
}GPU_Result;


using namespace std;
namespace{
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

    int kernel_height;
    int inner_node_size_wo_last_inner_node;
    int total;
    int total_per_gpu;

    Inner_node **d_innode;
    int **d_prefix;
    BLeaf_node **d_bleafnode;
    
    cudaStream_t **stream;

    float time_gpu = 0;

    //cudaEvent_t g_start,g_stop;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;

    int ngpus;


/*}}}*/
}


__global__ void full_multi_prefix_sort_1thread_notransfer_kernel(Inner_node *d_innode, BLeaf_node *d_bleafnode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count){
    /*{{{*/ 
    int key_base = Block_Dim / Thread_Per_Query * GPU_SWP_SIZE;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ Thread_Per_Query;  

    if(key_idx>=key_count) return;

    //int search_idx = 0; //threadIdx.x % PPI_Thread_Per_Query_1thread;
    int query_idx_in_local_block = threadIdx.x / Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    const int row = Block_Dim / Thread_Per_Query;//blockDim.x/ Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE;
    __shared__ int start_idx[row_swp];
    __shared__ int relist_idx[row_swp];

    //__shared__ char nexthalf[row];

    int stride = Block_Dim / Thread_Per_Query;
 
    for (int k = 0; k<Sort_Per_Thread; k++){
        start_idx[threadIdx.x + k* blockDim.x] = root_idx;
    }

    __syncthreads();

    key_t target;
    
    //without leaf
    for(int i=1;i<tree_height;i++){
        for(int j=0;j<GPU_SWP_SIZE;j++){
            // get key
            __syncthreads();

            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;
            int cur_r = r+stride *j;

            target = d_keys[cur_key];

            int pos = start_idx[cur_r];
            Inner_node *node = d_innode + pos;
            //search index;
            //key_t idx_key = node->inner_index[search_idx];
            int idx = 0;
            for(;idx<8;idx++){
                if(node->inner_index[idx] > target) break;
            }
            
            int begin = idx*8;

            idx = begin;
            for(;idx<begin+8;idx++){
                if(target < node->inner_key[idx]) break;
            }

            if(i==tree_height-1){
                start_idx[cur_r] = pos - inner_node_size_wo_last_inner_node;
                relist_idx[cur_r] = idx;

            }else 
                start_idx[cur_r] = __ldg(&d_prefix[pos]) + idx;
            __syncthreads();

        }
    }

   /* 
    for (int j=0; j<GPU_SWP_SIZE; j++) {
        __syncthreads();
            
        int cur_key = key_idx+stride *j;
        if(cur_key>=key_count)continue;
        int cur_r = r+stride *j;
            
        target = d_keys[cur_key];
        
        GPU_Result &result=d_gresult[cur_key];
        
        int pos = start_idx[cur_r];
        BLeaf_node *node = d_bleafnode + pos;
        char relist_id = relist_idx[cur_r]; 
        Record_list &relist = node->relist[relist_id];
        
        int idx = 0;
        for (; idx<4; idx++){
            if(relist.r[idx].r_key >= target) break;
        }
        if (relist.r[idx].r_key==target) {
            result.blfnode = pos; 
            //result.relist_idx = relist_id;
            //result.r_idx = idx;
            result.idx = ((idx & 3)<<6) | (relist_id & 63) ;
        }
        else result.blfnode = -1;
            
    }
    */
    
    
    
    
    
    /*}}}*/
}




/*--------------------------------------------------------------------------
*
*       full-multi-prefix-sort-1thread-notransfer
*       cpu dynamic
*
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
    }
 
    
    int stride = Para_Search_Bucket;        //这个版本的stride和其他版本的有所不同，其他版本的相当于这个stride * idx。
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        //11-13
        for(int i=0;i<total_per_gpu;i+=2){
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii]+stride*i, host_keys[iii]+stride*i, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][0]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii]+stride*(i+1), host_keys[iii]+stride*(i+1), batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][1]));
        }
       
        cudaEventRecord(g_start[iii]);
        for(int i=0;i<total_per_gpu;i++){
            int idx = i%2;

#ifdef TREE_32
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, d_keys[iii]+i*stride, d_keys_after[iii]+stride*idx, d_pos[iii]+stride*idx, d_pos_after[iii]+stride*idx, Para_Search_Bucket,16,32,stream[iii][idx]);
           
#else 
            cub::DeviceRadixSort::SortPairs(d_temp_storage[iii] + idx*temp_storage_bytes, temp_storage_bytes, d_keys[iii]+i*stride, d_keys_after[iii]+stride*idx, d_pos[iii]+stride*idx, d_pos_after[iii]+stride*idx, Para_Search_Bucket,48,64,stream[iii][idx]);
#endif  
            full_multi_prefix_sort_1thread_notransfer_kernel<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_bleafnode[iii],d_prefix[iii], inner_node_size_wo_last_inner_node, rootIdx, d_keys_after[iii]+stride*idx, kernel_height , d_gresult[iii]+stride*idx, Para_Search_Bucket);
        }

        cudaEventRecord(g_stop[iii]);
        
        
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii], d_gresult[iii], batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][0]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos[iii], d_pos_after[iii], batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][0]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+stride, d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][1]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos[iii]+stride, d_pos_after[iii]+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][1]));
    }
        
        
        
    return NULL;/*}}}*/
}

void full_multi_prefix_sort_1thread_notransfer(BPlusTree &tree,ifstream &search_file){
/*{{{*/

    //cudaGetDeviceCount(&ngpus);
    ngpus  = GPU_NUM;
    d_innode = (Inner_node **)malloc(ngpus * sizeof(Inner_node*));
    d_prefix = (int **)malloc(ngpus * sizeof(int*));
    d_bleafnode = (BLeaf_node **)malloc(ngpus * sizeof(BLeaf_node*));
    
    for (int i=0; i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        prepareGPU_v2(tree, d_innode[i], d_prefix[i]);
        prepareGPU_leaf(tree, d_bleafnode[i]);
    }

    rootIdx = tree.getRootIdx();
    inner_node_size_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();
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
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys[iii]), batch_d_key_size*total_per_gpu));
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
    

    cout<<"full multigpu prefix sort 1thread notransfer"<<endl;
    cout<<"GPU search total num:"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;

    pthread_join(tid, NULL);
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        cudaEventSynchronize(g_stop[i]);
          
        CUDA_ERROR_HANDLER(cudaEventElapsedTime(&time_gpu,g_start[i],g_stop[i]));
        cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    }

#ifdef ENABLE_TEST

    for (int iii=0;iii<ngpus;iii++) {
        for (int i=0;i<total_per_gpu;i++) {
            //int base = i*Para_Search_Bucket;
            int base = 0;
            for (int j=0; j<Para_Search_Bucket; j++) {
                int idx = base + j;
                key_t key = host_keys[iii][base + host_pos[iii][idx]];

                int blfnodeId = h_gresult[iii][idx].blfnode;
                int relist_idx = h_gresult[iii][idx].idx & 63;
                int r_idx = (h_gresult[iii][idx].idx >> 6) & 3;

                value_t v;
                if (blfnodeId==-1) v = -1;
                else v = tree.getLeafByIdx(blfnodeId)->relist[relist_idx].r[r_idx].val; 
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
        CUDA_ERROR_HANDLER(cudaFree(d_innode[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_prefix[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_bleafnode[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_after[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_temp_storage[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
    }

/*}}}*/

}


