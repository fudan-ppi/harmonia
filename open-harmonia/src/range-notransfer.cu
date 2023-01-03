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
#define Keys_Count 80*M

#define  GPU_SWP_SIZE 1 
//#define  GPU_SWP_SIZE (8*Thread_Per_Query) 


#define Block_Dim 16
#define Grid_Dim (8192*SM)


#define Thread_Per_Query 1
//#define Thread_Per_Query (DEFAULT_ORDER/Narrow_Rate) 
#define Sort_Per_Thread (GPU_SWP_SIZE / Thread_Per_Query)


#define Range_Size 8


typedef struct{
    key_t keys[Range_Size];
    value_t vals[Range_Size];
}GPU_Result;


using namespace std;
namespace{
/*{{{*/
    
    int Para_Search_Bucket = (Grid_Dim * Block_Dim) / Thread_Per_Query * GPU_SWP_SIZE;//851968

    dim3 block_dim(Block_Dim);
    dim3 grid_dim(Grid_Dim);

    int batch_d_key_size = sizeof(key_t) * Para_Search_Bucket;
    long batch_gresult_size = sizeof(GPU_Result) * Para_Search_Bucket;
    int batch_pos_size = sizeof(int) * Para_Search_Bucket;

    key_t **host_keys_begin;
    key_t **host_keys_end;
    int **host_pos; 
    GPU_Result  **h_gresult;
    
    key_t **d_keys_begin;
    key_t **d_keys_end;
    key_t **d_keys_begin_after;
    
    

    int **d_pos;
    int **d_pos_after;
    GPU_Result **d_gresult;
    
    
    //void *d_temp_storage = NULL;
    void **d_temp_storage;
    size_t temp_storage_bytes = 0;

    int rootIdx; 

    int kernel_height;
    int inner_node_size_wo_last_inner_node;
    int bleaf_count;
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

__global__ void range_prefix_sort_1thread_search_1_thread_scan_notransfer_kernel(Inner_node *d_innode, BLeaf_node *d_bleafnode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys_begin, key_t *d_keys_end,int *d_pos, int tree_height, GPU_Result *d_gresult, int key_count, int bleaf_count){
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

            target = d_keys_begin[cur_key];

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


    for (int j=0;j<GPU_SWP_SIZE;j++) {
        
        int cur_key = key_idx + stride * j;
        if (cur_key >= key_count) continue;
        int cur_r = r + stride * j;
        key_t start = d_keys_begin[cur_key]; 
        key_t end = d_keys_end[d_pos[cur_key]];
        //key_t end = d_keys_end[cur_key];
        GPU_Result &result=d_gresult[cur_key];
        
        int bleaf_pos = start_idx[cur_r];
        char relist_id = relist_idx[cur_r];
        char idx = 0;
       
        BLeaf_node *node = d_bleafnode + bleaf_pos;
        int used = node->used_relist_slot_num;

        bool start_flag = 0;     // when key is larger than start, flag = 1.
        int ans_num = 0;

        
        while (ans_num < Range_Size) {
            Record r = node->relist[relist_id].r[idx];
            //key_t key = r.r_key;
            //key_t key = node->relist[relist_id].r[idx].r_key; 

            if ( r.r_key == Max_Key ) {
                goto next_position2;
            }



            //if (start_flag == 0) {
            if (r.r_key < start) goto next_position;
            //    start_flag = 1;
            //}
            if (r.r_key > end) break;
            
            result.keys[ans_num] = r.r_key;
            result.vals[ans_num] = r.val;
            ans_num++;
            
next_position:
            idx++;
            if (idx == L_Fanout) {
next_position2:
                idx = 0;
                relist_id++;
                if (relist_id  >= used ) {
                    relist_id = 0;
                    bleaf_pos += 1;          // node's next; BLeaf_node is sorted.
                    if (bleaf_pos >= bleaf_count) break;
                    node += 1;              // node = d_bleafnode + bleaf_pos;
                    used = node->used_relist_slot_num;
                }
            }
        }
        

    }

    
    /*}}}*/
}




/*--------------------------------------------------------------------------
*
*       range
*       range-prefix-sort-1thread-search 
*       range-1thread-scan-notransfer
*       cpu dynamic
*
*
*-----------------------------------------------------------------------------*/

static void* launch_kernel_thread(void *args){
/*{{{*/
    
    g_start = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
    g_stop = (cudaEvent_t *)malloc(ngpus * sizeof(cudaEvent_t));
    

    
    for(int i=0;i<total_per_gpu;i++){
        for (int iii=0;iii<ngpus;iii++) {
            cudaSetDevice(iii+GPU_START);
            
            int idx = i%2;
            int stride = idx*Para_Search_Bucket;

            key_t *tmp_dkeys_begin = d_keys_begin[iii] + i * Para_Search_Bucket;
            key_t *tmp_dkeys_end = d_keys_end[iii] + i * Para_Search_Bucket;
            key_t *tmp_dkeys_begin_after = d_keys_begin_after[iii] + stride;
       
            key_t *tmp_hkeys_begin = host_keys_begin[iii] + i * Para_Search_Bucket; 
            key_t *tmp_hkeys_end = host_keys_end[iii] + i * Para_Search_Bucket; 
            
            
            
            
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(tmp_dkeys_begin, tmp_hkeys_begin, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][idx]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(tmp_dkeys_end, tmp_hkeys_end, batch_d_key_size, cudaMemcpyHostToDevice,stream[iii][idx]));
    
        }
    }
    
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

            key_t *tmp_dkeys_begin = d_keys_begin[iii] + i * Para_Search_Bucket;
            key_t *tmp_dkeys_end = d_keys_end[iii] + i * Para_Search_Bucket;
            key_t *tmp_dkeys_begin_after = d_keys_begin_after[iii] + stride;
            int *tmp_dpos =d_pos[iii] + stride;
            int *tmp_dpos_after =d_pos_after[iii] + stride;
            GPU_Result *tmp_d_gresult = d_gresult[iii] + stride;
            void *tmp_d_temp_storage = d_temp_storage[iii] + idx*temp_storage_bytes;
       
            key_t *tmp_hkeys_begin = host_keys_begin[iii] + i * Para_Search_Bucket; 
            key_t *tmp_hkeys_end = host_keys_end[iii] + i * Para_Search_Bucket; 
            GPU_Result *tmp_h_gresult = h_gresult[iii] + i * Para_Search_Bucket;
            int *tmp_hpos = host_pos[iii] + i * Para_Search_Bucket;
        
        
#ifdef TREE_32
            cub::DeviceRadixSort::SortPairs(tmp_d_temp_storage, temp_storage_bytes, tmp_dkeys_begin, tmp_dkeys_begin_after, tmp_dpos, tmp_dpos_after, Para_Search_Bucket,16,32,stream[iii][idx]);
#else         

            cub::DeviceRadixSort::SortPairs(tmp_d_temp_storage, temp_storage_bytes, tmp_dkeys_begin, tmp_dkeys_begin_after, tmp_dpos, tmp_dpos_after, Para_Search_Bucket,48,64,stream[iii][idx]);
#endif
            range_prefix_sort_1thread_search_1_thread_scan_notransfer_kernel<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_bleafnode[iii],d_prefix[iii], inner_node_size_wo_last_inner_node, rootIdx, tmp_dkeys_begin_after, tmp_dkeys_end,tmp_dpos_after,  kernel_height , tmp_d_gresult, Para_Search_Bucket, bleaf_count);
           

 /*          
            for (int j=0;j<Range_Size;j++) {
                int stride2 = Para_Search_Bucket / Range_Size; 
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(tmp_h_gresult+stride2*j, tmp_d_gresult+stride2*j, batch_gresult_size/Range_Size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            }
            
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(tmp_hpos, tmp_dpos_after, batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
    */        
        
        }
    }
        
        
        
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        cudaEventRecord(g_stop[iii]);
    }
    return NULL;/*}}}*/
}

void range_prefix_sort_1thread_search_and_1thread_scan_notransfer(BPlusTree &tree,ifstream &search_file){
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
    bleaf_count = tree.getBLeafSize();
    
    
    assert(rootIdx != -1);
   
    //for balance 
    //host_malloc
    host_keys_begin = (key_t **)malloc(ngpus * sizeof(key_t *));
    host_keys_end = (key_t **)malloc(ngpus * sizeof(key_t *));
    int nKeys_Count = Keys_Count / ngpus;
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_keys_begin[i]),sizeof(key_t)*nKeys_Count));
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_keys_end[i]),sizeof(key_t)*nKeys_Count));
    }
   
    //init key 
    int nums = 0;
    string s;
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys_begin[nums/nKeys_Count][nums%nKeys_Count] = key;
        getline(search_file,s);
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys_end[nums/nKeys_Count][nums%nKeys_Count] = Max_Key;
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
        CUDA_ERROR_HANDLER(cudaMallocHost(&(h_gresult[i]),long(sizeof(GPU_Result))*nKeys_Count));
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_pos[i]),sizeof(int)*nKeys_Count));
    }



    stream = (cudaStream_t **)malloc(ngpus * sizeof(cudaStream_t*));
    d_keys_begin = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_keys_end = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_keys_begin_after = (key_t **)malloc(ngpus * sizeof(key_t*));      //keys_begin after sorting 
    
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
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys_begin[iii]), batch_d_key_size*total_per_gpu));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys_end[iii]), batch_d_key_size*total_per_gpu));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys_begin_after[iii]), batch_d_key_size*2));
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
    cub::DeviceRadixSort::SortPairs(d_temp_storage[0], temp_storage_bytes, d_keys_begin[0], d_keys_begin_after[0], d_pos[0],d_pos_after[0],Para_Search_Bucket,16,32);
    
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_temp_storage[i]), temp_storage_bytes*2));
    }
    

    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
    

    cout<<"range: prefix-sort-1thread-search &  1thread-scan-nostransfer"<<endl;
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
            int base = i*Para_Search_Bucket;
            for (int j=0; j<Para_Search_Bucket; j++) {
                int idx = base + j;
                key_t key_begin = host_keys_begin[iii][base + host_pos[iii][idx]];
                key_t key_end = host_keys_end[iii][base + host_pos[iii][idx]]; 

                cout<<"start: "<<key_begin<<" end: "<<key_end<<endl;
                
                for (int jj = 0; jj<Range_Size;jj++) 
                    cout<<h_gresult[iii][idx].keys[jj]<<" : "<<h_gresult[iii][idx].vals[jj]<<endl;

            }
        }
    
    
    }
#endif

    for (int i=0;i<ngpus;i++) {
        
        cudaSetDevice(i+GPU_START);
        CUDA_ERROR_HANDLER(cudaFreeHost(h_gresult[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys_begin[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys_end[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_pos[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_innode[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_prefix[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_bleafnode[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_begin[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_end[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys_begin_after[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_temp_storage[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
    }

/*}}}*/

}


