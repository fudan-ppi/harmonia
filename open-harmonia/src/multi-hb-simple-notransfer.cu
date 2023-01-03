#include<cuda_runtime.h>
#include<iostream>
#include<assert.h>

#include"gbpt.h"
#include"cuda_utils.h"
#include<cuda_profiler_api.h>
#include"mempool.h"

#include<fstream>
#include<string>
#include<sys/time.h>
#include<omp.h>
#include"conf.h"
#define Thread_Per_Query 8
#define M 1000000
#define Keys_Count 200*M

#define Block_Dim 128
#define Grid_Dim (PPI_Block_Per_SM * SM *64)

typedef struct {
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;

using namespace std;

namespace{
/*{{{*/

    const int Para_Search_Bucket = Grid_Dim * Block_Dim / Thread_Per_Query;
    
    //int Para_Search_Bucket = (Grid_Dim * Block_Dim) / Thread_Per_Query * GPU_SWP_SIZE;
    // hb does not have GPU_SWP_SIZE. so we just increase the GridDim to maintain the bucket size
    
    
    //cudaStream_t stream[2];
    cudaStream_t **stream;
    
    dim3 block_dim(Block_Dim);
    dim3 grid_dim(Grid_Dim);

    //size:
    int d_keys_size = sizeof(key_t)*Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result)*Para_Search_Bucket;
    int d_cur_inner_node_size = sizeof(Inner_node *)*Para_Search_Bucket;

    
    //tree:
    int tree_height;
    Inner_node *root;
    Inner_node *inner_pool_start;
    
    
    
    //host:
    key_t **host_keys;
    GPU_Result **h_gresult;
    Inner_node ***cur_inner_node;
   
    //int rootIdx;


    //device:
    Inner_node **d_innode;
    key_t ** d_keys;
    GPU_Result **d_gresult;
    Inner_node ***d_cur_inner_node;


    const int total = Keys_Count / Para_Search_Bucket ;
    int total_per_gpu;




    float time_gpu = 0;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;

    int ngpus;
/*}}}*/
}

static __global__ void bpt_search_kernel_balance(Inner_node *d_innode,Inner_node ** d_cur_level_node/*当前搜索到的node*/,Inner_node *h_innode/*用于计算innernode idx*/,key_t *d_keys,int tree_height,GPU_Result *d_gresult,int key_count){    
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ Thread_Per_Query;/*{{{*/
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % Thread_Per_Query;
    int query_idx_in_local_block = threadIdx.x / Thread_Per_Query;
    const int r = query_idx_in_local_block;//just for simple

    key_t target = d_keys[key_idx];

    const int row = Block_Dim/ Thread_Per_Query;//blockDim.x/ Thread_Pre_Search;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = (long)(d_cur_level_node[key_idx] -  h_innode);//
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    
    GPU_Result &result = d_gresult[key_idx];

    for(int i=1;i<tree_height;i++){
        Inner_node *node = d_innode + start_idx[r];
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
                start_idx[r] = (long)node->child[begin+8];
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
                return;
            }else 
                start_idx[r]  = (long )node->child[idx]; 
        }
        inner_index_result[r] = 0;
        selfFlag = 0;
        flag[r][search_idx+1] = 0;
        __syncthreads();
    }/*}}}*/
}


//这里的本质是，把所有数据传到GPU上，在GPU上跑完一组存起来，后跑完的一组会覆盖前一组。
//所以在最后传回来的时候，传回来的是两个流分别做的最后一组。
//GPU上空间不够大，存不下所有的答案
static void* launch_kernel_thread(void *args) {
      /*{{{*/
        g_start = (cudaEvent_t*)malloc(ngpus * sizeof(cudaEvent_t));
        g_stop = (cudaEvent_t*)malloc(ngpus * sizeof(cudaEvent_t));

        for (int i=0;i<ngpus;i++) {
            cudaSetDevice(i+GPU_START);
            cudaEventCreate(&(g_start[i]));
            cudaEventCreate(&(g_stop[i]));
        }

    
        int stride = Para_Search_Bucket;        //这个版本的stride和其他版本的有所不同，其他版本的相当于这个stride * idx。
        
        
        for (int iii=0;iii<ngpus;iii++) {
            cudaSetDevice(iii+GPU_START);

            for(int i=0;i<total_per_gpu;i+=2){
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_cur_inner_node[iii]+stride*i,  cur_inner_node[iii]+stride*i, d_cur_inner_node_size,cudaMemcpyHostToDevice,stream[iii][0]));
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii]+stride*i,  host_keys[iii]+stride*i, d_keys_size,cudaMemcpyHostToDevice,stream[iii][0]));
            
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_cur_inner_node[iii]+stride*(i+1),  cur_inner_node[iii]+stride*(i+1), d_cur_inner_node_size,cudaMemcpyHostToDevice,stream[iii][1]));
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii]+stride*(i+1),  host_keys[iii]+stride*(i+1), d_keys_size,cudaMemcpyHostToDevice,stream[iii][1]));
            }              
       
            cudaEventRecord(g_start[iii]);
            for(int i=0;i<total_per_gpu;i++){
                int idx = i%2;
                bpt_search_kernel_balance<<<grid_dim,block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_cur_inner_node[iii]+i*stride,inner_pool_start,d_keys[iii]+i*stride,tree_height,d_gresult[iii]+idx*stride,Para_Search_Bucket);

            }
            cudaEventRecord(g_stop[iii]);

            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii],  d_gresult[iii], batch_gresult_size, cudaMemcpyDeviceToHost, stream[iii][0]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+stride,  d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost, stream[iii][1]));
            //这里实际上是只拷贝了最后两个bucket的答案回来，前面的都被覆盖掉了。。。。

        }
        for (int i=0;i<ngpus;i++) {
            cudaSetDevice(i+GPU_START);
        }
        return NULL;
    /*}}}*/
}





void HB_simple_notransfer(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    //cudaProfilerStart();

    //cudaGetDeviceCount(&ngpus);
    ngpus = GPU_NUM;
    d_innode = (Inner_node**)malloc(ngpus * sizeof(Inner_node*));

    for (int i=0;i<ngpus;i++) {
        CUDA_ERROR_HANDLER(cudaSetDevice(i+GPU_START));
        d_innode[i] = prepareGPU(tree);
    }

    //rootIdx = tree.getRootIdx();
    root = static_cast<Inner_node *>(tree.getRoot());

    unsigned int tmp_size;
    tree.getInnerSegementationInfo(inner_pool_start,tmp_size);

    tree_height = tree.getHeight();
    //assert(rootIdx != -1);


    host_keys = (key_t **)malloc(ngpus * sizeof(key_t *));
    cur_inner_node = (Inner_node ***)malloc(ngpus * sizeof(Inner_node **));
    h_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result *));

    int nKeys_Count = Keys_Count / ngpus;
    for (int i=0;i<ngpus;i++) {
        CUDA_ERROR_HANDLER(cudaSetDevice(i + GPU_START));
        //host malloc
        CUDA_ERROR_HANDLER(cudaMallocHost(&(host_keys[i]),sizeof(key_t)*nKeys_Count));//100M/ngpus numbers;
        CUDA_ERROR_HANDLER(cudaMallocHost(&(cur_inner_node[i]),sizeof(Inner_node *)*nKeys_Count));//100M/ngpus 
        CUDA_ERROR_HANDLER(cudaMallocHost(&(h_gresult[i]),sizeof(GPU_Result)*nKeys_Count));
    }
    
    
    int nums = 0;
    string s;
    while(getline(search_file,s)){
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums/nKeys_Count][nums%nKeys_Count] = key;
        nums++;
    }
    total_per_gpu = nKeys_Count / Para_Search_Bucket;

    stream = (cudaStream_t **)malloc(ngpus * sizeof(cudaStream_t*));
    d_keys = (key_t **)malloc(ngpus * sizeof(key_t*));
    d_gresult = (GPU_Result **)malloc(ngpus * sizeof(GPU_Result*));
    d_cur_inner_node = (Inner_node ***)malloc(ngpus * sizeof(Inner_node **)); 

    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        stream[iii] = (cudaStream_t *)malloc(2 * sizeof(cudaStream_t)); 
        for(int i=0;i<2;i++)    cudaStreamCreate(&(stream[iii][i]));

        //gpu malloc
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys[iii]),d_keys_size * total_per_gpu));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_gresult[iii]),batch_gresult_size * 2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_cur_inner_node[iii]), d_cur_inner_node_size* total_per_gpu));
        
    }
   
    
    
    //init cur_inner_node
    for (int iii=0;iii<ngpus;iii++) {
        for (int i=0; i<nKeys_Count; i++ ) {
            cur_inner_node[iii][i] = root;
        }
    }
    
    pthread_t tid;
    if ((pthread_create(&tid, NULL, launch_kernel_thread, NULL))!=0) {
        printf("can't create thread!\n");
   
    }
//TODO:     TEST!!!!! AND LAST LEVEL!!!

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    cout<<"HB+ tree no cpu simple version no transfer"<<endl;
    cout<<"GPU search total num:"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
    cout<<"total bucket num "<<total<<endl;
    
    pthread_join(tid, NULL);
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        cudaEventSynchronize(g_stop[i]);
          
        CUDA_ERROR_HANDLER(cudaEventElapsedTime(&time_gpu,g_start[i],g_stop[i]));
        cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    }

    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);

        CUDA_ERROR_HANDLER(cudaFree(d_innode[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(host_keys[i]));
        CUDA_ERROR_HANDLER(cudaFreeHost(cur_inner_node[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_keys[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_cur_inner_node[i]));
        CUDA_ERROR_HANDLER(cudaFree(d_gresult[i]));
        
        for(int j = 0;j<2;j++)
            cudaStreamDestroy(stream[i][j]);
    }


    //cudaProfilerStop();
/*}}}*/

}
