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

#define Thread_Pre_Search 8
#define M 1000000
#define Keys_Count  100*M
#define Block_Dim 128
#define Block_Pre_SM 1024

typedef struct {
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;

using namespace std;







__global__ void bpt_search_kernel(Inner_node *d_innode,int root_idx,key_t *d_keys,int tree_height,GPU_Result *d_gresult,int key_count){
    
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ Thread_Pre_Search;/*{{{*/
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % Thread_Pre_Search;
    int query_idx_in_local_block = threadIdx.x / Thread_Pre_Search;
    const int r = query_idx_in_local_block;//just for simple

    const int row = Block_Dim/ Thread_Pre_Search;//blockDim.x/ Thread_Pre_Search;
    __shared__ char flag[row][9];
    __shared__ int inner_index_result[row];
    __shared__ long  start_idx[row];
    
    start_idx[r] = root_idx;
    flag[r][search_idx] = 0;
    flag[r][search_idx+1] = 0;

    char selfFlag;
    __syncthreads();

    key_t target = d_keys[key_idx];

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



//basic
void BPT_Search_GPU(BPlusTree &tree,ifstream &search_file){
    Inner_node *d_innode = prepareGPU(tree);/*{{{*/

    int rootIdx = tree.getRootIdx();

    assert(rootIdx != -1);

    int Thread_Pre_Block = Block_Dim;
    int Block_Pre_Grid = 13 * Block_Pre_SM;//K80 has 13 SM
    const int Para_Search_Bucket = Block_Pre_Grid * Thread_Pre_Block / Thread_Pre_Search;

    dim3 block_dim(Thread_Pre_Block);
    dim3 grid_dim(Block_Pre_Grid);



    int d_keys_size = sizeof(key_t)*Para_Search_Bucket;

    //CPU malloc
    key_t *host_keys = (key_t *)malloc(Keys_Count * sizeof(key_t)); 
    GPU_Result h_gresult [Para_Search_Bucket];
  

    string s;
    int nums = 0;
    
    while(getline(search_file,s)) {
        key_t key;
        sscanf(s.c_str(),TYPE_D, &key);
        host_keys[nums++] = key;
    }



    //GPU malloc
    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult,sizeof(GPU_Result)*Para_Search_Bucket));
    key_t * d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys,d_keys_size));


    int total = nums / Para_Search_Bucket; 
    value_t val[Para_Search_Bucket];

    struct timeval start;
    struct timeval end;
    double t_transfer1=0;
    double t_gpu=0;
    double t_transfer2=0;
    double t_cpu=0;



    for (int i=0;i<total;i++) {
            // key transfer to GPU
            gettimeofday(&start, NULL); 
            CUDA_ERROR_HANDLER(cudaMemcpy(d_keys,host_keys + i*Para_Search_Bucket ,  d_keys_size,cudaMemcpyHostToDevice));
            gettimeofday(&end, NULL);

            t_transfer1 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
            

            //GPU search 
            gettimeofday(&start, NULL); 
            bpt_search_kernel<<<grid_dim,block_dim>>>(d_innode,rootIdx,d_keys,tree.getHeight(),d_gresult,Para_Search_Bucket);
            cudaDeviceSynchronize();
            gettimeofday(&end, NULL);
           
            t_gpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;

            
            gettimeofday(&start, NULL); 
            //result transfer back to CPU
            CUDA_ERROR_HANDLER(cudaMemcpy(h_gresult,d_gresult,sizeof(GPU_Result)*Para_Search_Bucket,cudaMemcpyDeviceToHost));
            gettimeofday(&end, NULL);
            t_transfer2 += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;


            value_t val[Para_Search_Bucket];  
            //CPU search 
            gettimeofday(&start, NULL); 

            int ss = i*Para_Search_Bucket;
            #pragma omp parallel for  
            for(int j=0;j<Para_Search_Bucket;j++){
                key_t key = host_keys[j+ss];
                BLeaf_node *blfnode = h_gresult[j].blfnode;
                val[j] = blfnode->findKey(h_gresult[j].relist_idx,key);
            }

            //test
            
            //for (int j=0;j<Para_Search_Bucket;j++) {
            //    key_t key = host_keys[j+ss];
            //    cout<<key<<": "<<val[j]<<endl;
            //}



            gettimeofday(&end, NULL);
            t_cpu += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    }
    cout<<"HB+ tree GPU [serial]:"<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"t_transfer1:         "<<t_transfer1<<endl;
    cout<<"t_gpu:               "<<t_gpu<<endl;
    cout<<"t_transfer2:         "<<t_transfer2<<endl;
    cout<<"t_cpu:               "<<t_cpu<<endl;
    cout<<"total time:          "<<t_transfer1+t_gpu+t_transfer2+t_cpu<<endl;;
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));/*}}}*/
}

void BPT_Search_GPU_DoubleBuffering(BPlusTree &tree,ifstream &search_file){
     /*{{{*/
    Inner_node *d_innode = prepareGPU(tree);
    int rootIdx = tree.getRootIdx();

    assert(rootIdx != -1);

    int Thread_Pre_Block = Block_Dim;
    int Block_Pre_Grid = 13 * Block_Pre_SM;//K80 has 13 SM
    const int Para_Search_Bucket = Block_Pre_Grid * Thread_Pre_Block / Thread_Pre_Search;

    dim3 block_dim(Thread_Pre_Block);
    dim3 grid_dim(Block_Pre_Grid);

    key_t *host_keys;
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));//100M numbers;
    int nums = 0;
    string s;
    while(getline(search_file,s)){
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    }

    GPU_Result *h_gresult ;
    int batch_gresult_size = sizeof(GPU_Result)*Para_Search_Bucket;
    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,batch_gresult_size *2));

    cudaStream_t stream[2];
    for(int i=0;i<2;i++){
        cudaStreamCreate(&stream[i]);
    }

    key_t * d_keys;
    int d_keys_size = sizeof(key_t)*Para_Search_Bucket;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys,d_keys_size *2));

    GPU_Result *d_gresult;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult,batch_gresult_size * 2));

    struct timeval start;
    struct timeval end;
    double total_time=0;

    int total = Keys_Count / Para_Search_Bucket ;

    value_t val[Para_Search_Bucket];  
    for(int i=0;i<2&&i<total;i++){
       CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + i*Para_Search_Bucket, host_keys + i*Para_Search_Bucket, d_keys_size,cudaMemcpyHostToDevice,stream[i]));
       bpt_search_kernel<<<grid_dim,block_dim,0,stream[i]>>>(d_innode,rootIdx,d_keys+i*Para_Search_Bucket,tree.getHeight(),d_gresult+i*Para_Search_Bucket,Para_Search_Bucket);
    }

    gettimeofday(&start, NULL); 
    int idx = 2;
    while(idx<total+2){
        int i = idx%2;
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket,d_gresult+i*Para_Search_Bucket,batch_gresult_size,cudaMemcpyDeviceToHost,stream[i]));
        cudaStreamSynchronize(stream[i]);

        if(idx<total){
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + i * Para_Search_Bucket, host_keys + idx*Para_Search_Bucket,d_keys_size,cudaMemcpyHostToDevice,stream[i]));
            bpt_search_kernel<<<grid_dim,block_dim,0,stream[i]>>>(d_innode,rootIdx,d_keys+i*Para_Search_Bucket,tree.getHeight(),d_gresult+i*Para_Search_Bucket,Para_Search_Bucket);
        }
        //do cpu
        int start = (idx-2)*Para_Search_Bucket;
        int node_start = i*Para_Search_Bucket;
        #pragma omp parallel for  
        for(int j=0;j<Para_Search_Bucket;j++){
            key_t key = host_keys[start+j];
            BLeaf_node *blfnode =  h_gresult[node_start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[j].relist_idx,key);
             //if(val1 == -1)continue;
             //if(key!= val1)     cout<<key<<": "<<val1<<endl;
        }
        
        idx++;
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    cout<<"HB+ tree [double buffer]"<<endl;
    cout<<"Search count"<<total*Para_Search_Bucket<<endl;
    cout<<"GPU double_buffering total time:          "<<total_time<<endl;;
        // search GPU for rest of key 
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

    for(int i = 0;i<2;++i)
        cudaStreamDestroy(stream[i]);

/*}}}*/
}


/*-----------------------------------------------------------------------------------
*
*
*
*           GPU_V2
*
*
*
*----------------------------------------------------------------------------------*/

volatile  int stream_flag[2] = {-1,-1};
    
int Thread_Pre_Block = Block_Dim;
int Block_Pre_Grid = 13 * Block_Pre_SM;//K80 has 13 SM
const int Para_Search_Bucket = Block_Pre_Grid * Thread_Pre_Block / Thread_Pre_Search;

int d_keys_size = sizeof(key_t)*Para_Search_Bucket;
int batch_gresult_size = sizeof(GPU_Result)*Para_Search_Bucket;

int rootIdx;
int tree_height;

key_t *host_keys;
key_t * d_keys;
GPU_Result *h_gresult ;
GPU_Result *d_gresult;

cudaStream_t stream[2];
dim3 block_dim(Thread_Pre_Block);
dim3 grid_dim(Block_Pre_Grid);

Inner_node *d_innode;

const int total = Keys_Count / Para_Search_Bucket ;

void CUDART_CB CallBack(cudaStream_t stream, cudaError_t status, void *data) {
    stream_flag[(size_t)data%2] = (size_t)data; 
}



cudaEvent_t g_start,g_stop;

void* launch_kernel(void *args) {
    /*{{{*/
        cudaEventRecord(g_start);
        for(int i=0;i<total;i++){
            int idx = i%2;
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + idx*Para_Search_Bucket,  host_keys + i*Para_Search_Bucket, d_keys_size,cudaMemcpyHostToDevice,stream[idx]));
            bpt_search_kernel<<<grid_dim,block_dim,0,stream[idx]>>>(d_innode,rootIdx,d_keys+idx*Para_Search_Bucket,tree_height,d_gresult+idx*Para_Search_Bucket,Para_Search_Bucket);
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult + i*Para_Search_Bucket,  d_gresult + idx*Para_Search_Bucket, batch_gresult_size, cudaMemcpyDeviceToHost, stream[idx]));
            cudaStreamAddCallback(stream[idx], CallBack, (void *)i, 0);
        }
        cudaEventRecord(g_stop);
    return NULL;
    /*}}}*/
}


void BPT_Search_GPU_DoubleBuffering_v2(BPlusTree &tree,ifstream &search_file){
    /*{{{*/
    d_innode = prepareGPU(tree);
    rootIdx = tree.getRootIdx();
    tree_height = tree.getHeight();

    assert(rootIdx != -1);

    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));//100M numbers;
    int nums = 0;
    string s;
    while(getline(search_file,s)){
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    }

    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*Keys_Count));

    for(int i=0;i<2;i++){
        cudaStreamCreate(&stream[i]);
    }

    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys,d_keys_size *2));
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult,batch_gresult_size * 2));

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time=0;


    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);

    pthread_t ntid;
    if ((pthread_create(&ntid, NULL, launch_kernel, NULL))!=0) {
        printf("can't create thread!\n");
    }

    gettimeofday(&start, NULL); 
    //cout<<ntid<<endl; 
    value_t val[Para_Search_Bucket]; 
    double tmp_time = 0;
    //#pragma omp parallel for  
    for (int idx=0; idx<total;idx++){
        int i = idx%2;
        //do cpu
        int start = idx*Para_Search_Bucket;
        while(stream_flag[i]<idx) ;

    gettimeofday(&start1, NULL); 
        #pragma omp parallel for  
        for(int j=0;j<Para_Search_Bucket;j++){
            key_t key = host_keys[start+j];
            BLeaf_node *blfnode =  h_gresult[start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[start+j].relist_idx,key);
            //if(val[j] == -1)continue;
            //cout<<key<<": "<<val[j]<<endl;
            //if(key!= val1)     cout<<key<<": "<<val1<<endl;
        }  
    gettimeofday(&end1, NULL); 

    tmp_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
        //test
        /*        
        for (int j=0;j<Para_Search_Bucket;j++) {
            key_t key = host_keys[start+j];
            cout<<key<<": "<<val[j]<<endl;
        }
        */
    }


    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    
    
    
    
    
    cudaEventSynchronize(g_stop);
    float g_time = 0;
    cudaEventElapsedTime(&g_time,g_start,g_stop);
    
    cout<<"HB+ tree [double buffer two thread]"<<endl;
    cout<<"Search count "<<total*Para_Search_Bucket<<endl;
    cout<<"total bucket num "<<total<<endl;
    cout<<"CPU time "<<tmp_time<<endl;
    cout<<"GPU time  "<<g_time<<endl;
    cout<<"GPU double_buffering total time:          "<<total_time<<endl;;
        // search GPU for rest of key 
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

    for(int i = 0;i<2;++i)
        cudaStreamDestroy(stream[i]);

     /*}}}*/

}
/*-----------------------------------------------------------------------------------
*
*
*
*           GPU_V3
*
*
*
*----------------------------------------------------------------------------------*/
volatile int key_status = -1;

int level = 1;
Inner_node **cur_inner_node;
Inner_node *root;
Inner_node *inner_pool_start;
Inner_node **d_cur_inner_node;
int d_cur_inner_node_size = sizeof(Inner_node *)*Para_Search_Bucket;




__global__ void bpt_search_kernel_balance(Inner_node *d_innode,Inner_node ** d_cur_level_node/*当前搜索到的node*/,Inner_node *h_innode/*用于计算innernode idx*/,key_t *d_keys,int tree_height,GPU_Result *d_gresult,int key_count){    
    int key_idx = (blockIdx.x * blockDim.x + threadIdx.x)/ Thread_Pre_Search;/*{{{*/
    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % Thread_Pre_Search;
    int query_idx_in_local_block = threadIdx.x / Thread_Pre_Search;
    const int r = query_idx_in_local_block;//just for simple

    key_t target = d_keys[key_idx];

    const int row = Block_Dim/ Thread_Pre_Search;//blockDim.x/ Thread_Pre_Search;
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

void * load_balance_CPU(void *  arg){// similiar to search_swp
    assert(level<tree_height);/*{{{*/
    struct timeval start;
    struct timeval end;

    int idx = 0;
    const int count = Para_Search_Bucket /P;
    gettimeofday(&start, NULL); 
    while(idx<total){
        int start = idx*Para_Search_Bucket;
        omp_set_num_threads(12);
        #pragma omp parallel for  
        for(int i=0;i<count;i++){
            vector<Inner_node *> nodes(P,root);
            int relist_idx[P];
            for(int step = 1;step <=level;step++){
                for(int k = 0;k<P;k++){
                   relist_idx[k] = nodes[k]->getChildIdx_avx2(NULL,host_keys[start+i*P+k]);
                   nodes[k] = static_cast<Inner_node *>(((Inner_node *)nodes[k])->child[relist_idx[k]]);
                   __builtin_prefetch(nodes[k],0,3);
                }
            }
            for(int k=0;k<P;k++) cur_inner_node[start+i*P+k] = nodes[k];
        }
        key_status++;
        idx++;
    }
    gettimeofday(&end, NULL);
    cout<<"CPU load balance"<<(end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0<<endl;
    return NULL;/*}}}*/
}
void* launch_kernel_balance(void *args) {
        cudaEvent_t start,stop;/*{{{*/
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for(int i=0;i<total;i++){
            while(key_status<i);//wait for load balance cpu 

            int idx = i%2;
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_cur_inner_node + idx*Para_Search_Bucket,  cur_inner_node + i*Para_Search_Bucket, d_cur_inner_node_size,cudaMemcpyHostToDevice,stream[idx]));

            CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + idx*Para_Search_Bucket,  host_keys + i*Para_Search_Bucket, d_keys_size,cudaMemcpyHostToDevice,stream[idx]));
            bpt_search_kernel_balance<<<grid_dim,block_dim,0,stream[idx]>>>(d_innode,d_cur_inner_node + idx*Para_Search_Bucket,inner_pool_start,d_keys+idx*Para_Search_Bucket,tree_height-level,d_gresult+idx*Para_Search_Bucket,Para_Search_Bucket);
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult + i*Para_Search_Bucket,  d_gresult + idx*Para_Search_Bucket, batch_gresult_size, cudaMemcpyDeviceToHost, stream[idx]));
            cudaStreamAddCallback(stream[idx], CallBack, (void *)i, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time,start,stop);
        cout<<"GPU time  "<<time<<endl;
    return NULL;
    /*}}}*/
}

void* launch_kernel_balance_v2(void *args) {
/*{{{*/
    cudaEvent_t start[total],stop[total];
        
    for (int i=0;i<total;i++) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }
    for(int i=0;i<total;i++){
        while(key_status<i);//wait for load balance cpu 

        int idx = i%2;

        cudaEventRecord(start[i], stream[idx]);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_cur_inner_node + idx*Para_Search_Bucket,  cur_inner_node + i*Para_Search_Bucket, d_cur_inner_node_size,cudaMemcpyHostToDevice,stream[idx]));

        CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys + idx*Para_Search_Bucket,  host_keys + i*Para_Search_Bucket, d_keys_size,cudaMemcpyHostToDevice,stream[idx]));
        bpt_search_kernel_balance<<<grid_dim,block_dim,0,stream[idx]>>>(d_innode,d_cur_inner_node + idx*Para_Search_Bucket,inner_pool_start,d_keys+idx*Para_Search_Bucket,tree_height-level,d_gresult+idx*Para_Search_Bucket,Para_Search_Bucket);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult + i*Para_Search_Bucket,  d_gresult + idx*Para_Search_Bucket, batch_gresult_size, cudaMemcpyDeviceToHost, stream[idx]));
        cudaStreamAddCallback(stream[idx], CallBack, (void *)i, 0);
        cudaEventRecord(stop[i], stream[idx]);
    }
    
    float time = 0;
    
    for (int i=0;i<total;i+=2) {    
        float time1 = 0;
        cudaEventSynchronize(stop[i]);
        cudaEventElapsedTime(&time1,start[i],stop[i]);
    
        time += time1; 
    }
    cout<<"GPU time  "<<time<<endl;
    return NULL;
/*}}}*/
    
}




void BPT_Search_GPU_DoubleBuffering_v3(BPlusTree &tree,ifstream &search_file){
    cudaProfilerStart();/*{{{*/
    d_innode = prepareGPU(tree);
    rootIdx = tree.getRootIdx();
    root = static_cast<Inner_node *>(tree.getRoot());

    unsigned int tmp_size;
    tree.getInnerSegementationInfo(inner_pool_start,tmp_size);

    tree_height = tree.getHeight();
    assert(rootIdx != -1);

    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Keys_Count));//100M numbers;
    CUDA_ERROR_HANDLER(cudaMallocHost(&cur_inner_node,sizeof(Inner_node *)*Keys_Count));//100M 

    int nums = 0;
    string s;
    while(getline(search_file,s)){
        key_t key;
        sscanf(s.c_str(),TYPE_D,&key);
        host_keys[nums++] = key;
    }



    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*Keys_Count));

    for(int i=0;i<2;i++){
        cudaStreamCreate(&stream[i]);
    }

    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys,d_keys_size *2));
    CUDA_ERROR_HANDLER(cudaMalloc(&d_gresult,batch_gresult_size * 2));
    CUDA_ERROR_HANDLER(cudaMalloc(&d_cur_inner_node, d_cur_inner_node_size*2));

    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time=0;

    gettimeofday(&start, NULL); 
    //load_balance_CPU(NULL);
    pthread_t ntid;
    if ((pthread_create(&ntid,NULL, load_balance_CPU, NULL))!=0) {
        printf("can't create thread!\n");
    }
    stream_flag[0] = -1;
    stream_flag[1] = -1;

    pthread_t ntid1;
    if ((pthread_create(&ntid1, NULL, launch_kernel_balance, NULL))!=0) {
        printf("can't create thread!\n");
    }

    value_t val[Para_Search_Bucket]; 
    double tmp_time = 0;
    for (int idx=0; idx<total;idx++){
        int i = idx%2;
        //do cpu
        int start = idx*Para_Search_Bucket;
        while(stream_flag[i]<idx) ;

        gettimeofday(&start1, NULL); 
        if(key_status<total-1)
            omp_set_num_threads(40);
        else omp_set_num_threads(52);
        #pragma omp parallel for  
        for(int j=0;j<Para_Search_Bucket;j++){
            key_t key = host_keys[start+j];
            BLeaf_node *blfnode =  h_gresult[start+j].blfnode;
            val[j] = blfnode->findKey(h_gresult[start+j].relist_idx,key);
            //if(val[j] == -1)continue;
            //cout<<key<<": "<<val[j]<<endl;
            //if(key!= val1)     cout<<key<<": "<<val1<<endl;
        }  
    gettimeofday(&end1, NULL); 

    tmp_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
        //test
        /*        
        for (int j=0;j<Para_Search_Bucket;j++) {
            key_t key = host_keys[start+j];
            cout<<key<<": "<<val[j]<<endl;
        }
        */
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    cout<<"HB+ tree [double buffer load balance]"<<endl;
    cout<<"Search count "<<total*Para_Search_Bucket<<endl;
    cout<<"total bucket num "<<total<<endl;
    cout<<"CPU time "<<tmp_time<<endl;
    cout<<"GPU double_buffering total time:          "<<total_time<<endl;;
        // search GPU for rest of key 
    
    CUDA_ERROR_HANDLER(cudaFree(d_innode));
    CUDA_ERROR_HANDLER(cudaFreeHost(host_keys));
    CUDA_ERROR_HANDLER(cudaFreeHost(cur_inner_node));
    CUDA_ERROR_HANDLER(cudaFree(d_keys));
    CUDA_ERROR_HANDLER(cudaFree(d_cur_inner_node));
    CUDA_ERROR_HANDLER(cudaFree(d_gresult));

    for(int i = 0;i<2;++i)
        cudaStreamDestroy(stream[i]);

     cudaProfilerStop();/*}}}*/

}

