#include<cuda_runtime.h>
#include<iostream>
#include<sys/time.h>
#include <assert.h>

#include <string>
#include"ppi-bpt.h"
#include"cuda_utils.h"
#include"mempool.h"
#include"buffer.h"
#include<sched.h>
#include<pthread.h>
#include<unistd.h>

#include <fstream>
#include <omp.h>
#include "cub/cub.cuh"
#include "conf.h"
#define M 1000000
#define Keys_Count 100*M

#define  GPU_SWP_SIZE4 (16*2) 


#define PPI_Block_Dim_4thread 128
//#define PPI_Block_Per_SM 64

#define PPI_Thread_Per_Query_4thread 4
#define  Sort_Per_4Thread (GPU_SWP_SIZE4 / PPI_Thread_Per_Query_4thread)

//#define CPU_THREAD 54 


#define LOAD_BALANCE_LEVEL 1 //CPU maybe compute last N level inner_node
#define LOAD_BALANCE_PECENTAGE 0 //unit %; Range 0~1 ,0 means CPU compute 0% inner_node,100% means CPU compute compute all inner_node

//thread scheduling 
#define PROCESS_CHRUNK_SIZE 1024  
#define DISPATCHER_THREAD_NUM 1

//#define SPLIT_BUFFER_NUM (PROCESS_THREAD_NUM / 4)
#define SPLIT_BUFFER_NUM 2






typedef struct{
    int blfnode;
    char relist_idx;
}GPU_Result;

typedef struct {
    int bufferIdx;
    int workload;
}BufferWorkLeft;


using namespace std;
namespace{
/*{{{*/
    
    volatile int** stream_flag;
    //volatile int stream_flag[2] = {-1,-1};

    int Thread_Per_Block = PPI_Block_Dim_4thread;
    int Block_Per_Grid = SM * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (Block_Per_Grid * Thread_Per_Block) / PPI_Thread_Per_Query_4thread * GPU_SWP_SIZE4;//851968

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
    
    
    //cudaStream_t stream[2];
    cudaStream_t **stream;

/*
    void CUDART_CB CallBack(cudaStream_t stream,cudaError_t status,void *data){
        stream_flag[(size_t)data%2] = (size_t)data;
    }
*/
    void CUDART_CB CallBack(cudaStream_t stream,cudaError_t status,void *data){
        stream_flag[(size_t)data/total_per_gpu][(size_t)data%total_per_gpu%2] = (size_t)data;
    }

    float time_gpu = 0;

    //cudaEvent_t g_start,g_stop;
    cudaEvent_t *g_start;
    cudaEvent_t *g_stop;


    //---
    volatile int key_status = -1;

    BPlusTree *bptree;
    //vector<double> cpu_thread_total_time(CPU_THREAD,0);
    //vector<double> cpu_thread_compute_time(CPU_THREAD,0);
   
    //load balance 
    int bucket_balance_keys = ((int)((Para_Search_Bucket * LOAD_BALANCE_PECENTAGE+1024-1)/ 1024)) * 1024;//1024：the num of key which Theadblock process

    //thread scheduling
    key_t *range;
    vector<LBBuffer> buffers(SPLIT_BUFFER_NUM);
    vector<Thread_To_LBBuffer> thread_buffer_mapping(PROCESS_THREAD_NUM);


    int ngpus;


/*}}}*/
}



__global__ void multi_prefix_sort_4thread_kernel(Inner_node *d_innode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count,int d_bucket_balance_keys){
    /*{{{*/ 
    int key_base = PPI_Block_Dim_4thread / PPI_Thread_Per_Query_4thread * GPU_SWP_SIZE4;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_4thread;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_4thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_4thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_4thread/ PPI_Thread_Per_Query_4thread;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE4;
    __shared__ char flag[row][5];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = PPI_Block_Dim_4thread / PPI_Thread_Per_Query_4thread;
    
    /*
    for(int i=0;i<GPU_SWP_SIZE2;i++){
        int cur_key = key_idx+stride*i;
        if(cur_key>=key_count)continue;
        start_idx[r + stride*i] = d_inter_result[cur_key];
    }
*/
 
    for (int k = 0; k<Sort_Per_4Thread; k++){
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
        for(int j=0;j<GPU_SWP_SIZE4;j++){
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
                idx_key = node->inner_index[search_idx+4];
                if(target < idx_key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                //__syncthreads();
                __threadfence_block();
            }

            if(selfFlag == 1 && flag[r][search_idx] == 0){
                inner_index_result[r] = search_idx+nexthalf[r]*4; 
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
                key = node->inner_key[idx+4];
                if(target < key){
                    flag[r][search_idx+1] = 1;
                    selfFlag = 1;
                    nexthalf[r] = 1;
                }
                //__syncthreads();
                __threadfence_block();
            }

     
            
            //get next child;
            if(selfFlag == 1 && flag[r][search_idx] == 0){
                if(i==tree_height-1){
                    if(isBalance){
                        result.blfnode = __ldg(&d_prefix[pos])+idx+nexthalf[r]*4 ;
                        result.relist_idx = 111;
                   }else{ 
                        //result.blfnode = pos - inner_node_size_wo_last_inner_node;//useless
                        result.relist_idx = idx + nexthalf[r]*4;
                   }    

                }else 
                    start_idx[cur_r] = __ldg(&d_prefix[pos]) + (idx + nexthalf[r]*4);
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
*       multi-prefix-sort-4thread 
*       cpu dynamic
*
*
*-----------------------------------------------------------------------------*/

static void* launch_kernel_thread(void *args){
/*{{{*/
    stick_this_thread_to_core2(1);
    
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
        
            multi_prefix_sort_4thread_kernel<<<grid_dim, block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_prefix[iii], inner_node_size_wo_last_inner_node, rootIdx, d_keys_after[iii]+stride, kernel_height , d_gresult[iii]+stride, Para_Search_Bucket, bucket_balance_keys);
            
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii]+i*Para_Search_Bucket, d_gresult[iii]+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos[iii]+i*Para_Search_Bucket, d_pos_after[iii]+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[iii][idx]));
            
            cudaStreamAddCallback(stream[iii][idx],CallBack,(void *)(i+iii*total_per_gpu),0);
        
        }
    }
        
        
        
    for (int iii=0;iii<ngpus;iii++) {
        cudaSetDevice(iii+GPU_START);
        cudaEventRecord(g_stop[iii]);
    }
    return NULL;/*}}}*/
}

static void * dispatcher(void *args){
 /*{{{*/
    stick_this_thread_to_core2(2);

    vector<BufferWorkLeft> workLeftVector(SPLIT_BUFFER_NUM) ;


    for (int i=0;i<total_per_gpu;i++) {
    
        int idx = i%2;
        int start = i * Para_Search_Bucket; 
        for (int iii=0;iii<ngpus;iii++) {
            while (stream_flag[iii][idx]<i+iii*total_per_gpu); 
      
            int cur_idx = 0;

            for (int j=0; j<Para_Search_Bucket; j+=PROCESS_CHRUNK_SIZE){
                int idx = start + j;
                key_t key = host_keys[iii][start + host_pos[iii][idx]];
                
                if(cur_idx < SPLIT_BUFFER_NUM-1 && key >= range[cur_idx]){
                    cur_idx++;
                }

                buffers[cur_idx].put(idx, idx + PROCESS_CHRUNK_SIZE - 1,i+iii*total_per_gpu);
            
            }
        
        }


        
        /* for (auto aaa:buffers) {
            cout<<"size: "<<aaa.buffer_queue.size()<<endl; 
        }*/

        //thread_adjustment

        if (i%6==5) {        
        //if ((i>total / 8)&&(i%8==0)) {        
            //cout<<"i: "<<i<<endl;
            for (int j = 0; j<SPLIT_BUFFER_NUM; j++) {
                //cout<<j<<"  ";
                workLeftVector[j].bufferIdx = j;
                workLeftVector[j].workload  = buffers[j].getWorkLeft(); 
            }
/*
            sort(workLeftVector.begin(), workLeftVector.end(),
                    [](const BufferWorkLeft &b1, const BufferWorkLeft &b2){
                        return b1.workload < b2.workload;
                    });

            BufferWorkLeft front = workLeftVector.front();
            BufferWorkLeft back = workLeftVector.back();
*/
            int maxx = 0;
            int minn = 0;
            for (int j=1;j<SPLIT_BUFFER_NUM;j++) {
               if (workLeftVector[j].workload > workLeftVector[maxx].workload) maxx = j;
               else if (workLeftVector[j].workload < workLeftVector[minn].workload) minn = j;
            }


            BufferWorkLeft front = workLeftVector[minn];
            BufferWorkLeft back = workLeftVector[maxx];

            if ((front.workload * 2 < back.workload)&&(buffers[front.bufferIdx].getThreadNum()>=4)) {
                for (int jj=0;jj<2;jj++) {
                    Thread_To_LBBuffer * tp = buffers[front.bufferIdx].popThread();
                    tp->setMappingAndProcessIdx(back.bufferIdx ,buffers[back.bufferIdx].getNextProcessIdx());
                    buffers[back.bufferIdx].pushThread(tp);
                    //cout<<"sche: thread: from "<<front.bufferIdx<<" to "<<back.bufferIdx<<" at "<<i<<endl;
                }
            }

            //cout<<"-----------------"<<endl;
        }
 


    }


    //cout<<"finish"<<endl;
    for(int i=0;i<SPLIT_BUFFER_NUM;i++){
        buffers[i].setFinish();
    }
/*}}}*/
}


static void *processer(void *args){
/*{{{*/
    int threadID = (long long)args;
    stick_this_thread_to_core2(threadID+2+2);
        
    //vector<void *> nodes(PROCESS_CHRUNK_SIZE);
    //vector<int> relist_idxs(PROCESS_CHRUNK_SIZE);
    vector<key_t> vals(PROCESS_CHRUNK_SIZE);

    while(1){
        std::pair<int,int> pair = thread_buffer_mapping[threadID].getMappingAndProcessIdx();
        int buff_idx = pair.first;
        int idx = pair.second;

        tuple<int,int,int> range = buffers[buff_idx].get(idx);
        thread_buffer_mapping[threadID].setProcessIdx(buff_idx, idx);//update idx 

        int start = get<0>(range);
        int end = get<1>(range);
        int bucket_idx_whole= get<2>(range);

        int bucket_idx = bucket_idx_whole % total_per_gpu;
        int gpu_idx = bucket_idx / total_per_gpu;

        int bucket_base = bucket_idx * Para_Search_Bucket;
       




        if(start == -1) {
            return NULL;
        }

        for(int k = start,j=0;k <=end;k++,j++){ 

            key_t key = host_keys[gpu_idx][bucket_base + host_pos[gpu_idx][k]];
            
            if(h_gresult[gpu_idx][k].relist_idx == 111){ 
                
                void *node = (void *)(bptree->getInnerNodeByIdx(h_gresult[gpu_idx][k].blfnode));

                int relist_idx = 0;
                
                for(int i=1;i<=LOAD_BALANCE_LEVEL;i++){
                    relist_idx = ((Inner_node *)node)->getChildIdx_avx2(NULL,key);
                    node = ((Inner_node *)node)->child[relist_idx];
                    //__builtin_prefetch(node,0,3);
                }
                vals[j]=((BLeaf_node *)node)->findKey(relist_idx,key);
            }else{
                BLeaf_node *blfnode = bptree->getLeafByIdx(h_gresult[gpu_idx][k].blfnode);
                vals[j] = blfnode->findKey(h_gresult[gpu_idx][k].relist_idx,key);
            } 
        }

        /*
        //CPU process bleaf_node
        for(int k = start,j=0;k <=end;k++,j++){
            if(h_gresult[k].relist_idx != 111) break;//1024的倍数
            key_t key = host_keys[bucket_base + host_pos[k]];
            vals[j]=((BLeaf_node *)nodes[j])->findKey(relist_idxs[j],key);
        }
        */
        //test
#ifdef ENABLE_TEST
        for(int k =start,j=0;k <=end;k++,j++){
            key_t key = host_keys[gpu_idx][bucket_base + host_pos[gpu_idx][k]];
            //if(vals[j]!=key)
                //cout<<threadID<<" "<<key<<" "<<vals[j]<<endl;
            printf("%d %lld %lld\n", threadID, key, vals[j]);
        }
#endif
        
    }
/*}}}*/
}


void multi_prefix_sort_4thread(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    stick_this_thread_to_core2(0);

    //cudaGetDeviceCount(&ngpus);
    ngpus  = GPU_NUM;
    bptree = &tree;
    
    
    d_innode = (Inner_node **)malloc(ngpus * sizeof(Inner_node*));
    d_prefix = (int **)malloc(ngpus * sizeof(int*));
    
    for (int i=0; i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        prepareGPU_v2(tree, d_innode[i], d_prefix[i]);
    }

//for thread scheduling 
    range = (key_t *)malloc((SPLIT_BUFFER_NUM-1)*sizeof(key_t));
    tree.getKeyRangeOfInnerNode(tree.getHeight()-1-LOAD_BALANCE_LEVEL, SPLIT_BUFFER_NUM, range);

    for(int i=0;i<PROCESS_THREAD_NUM;i++){

        //thread_buffer_mapping[i].setMapping(i/4);//init 4 thread per buffer;
        //buffers[i/4].pushThread(i);
#ifdef thread_hyper
        thread_buffer_mapping[i].setMappingAndProcessIdx(i/2%SPLIT_BUFFER_NUM,0);
        buffers[i/2%SPLIT_BUFFER_NUM].pushThread( &thread_buffer_mapping[i]);
#else
        thread_buffer_mapping[i].setMappingAndProcessIdx(i%SPLIT_BUFFER_NUM,0);
        buffers[i%SPLIT_BUFFER_NUM].pushThread(&thread_buffer_mapping[i]);
#endif
    }
// 

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



    //cout<<bucket_balance_keys<<endl;
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
    
    stream_flag = (volatile int **)malloc(ngpus * sizeof(int*)); 
    for (int i=0;i<ngpus;i++) {
        stream_flag[i] = (volatile int*)malloc(2*sizeof(int));
        stream_flag[i][0] = -1;
        stream_flag[i][1] = -1;
    }


    pthread_t tid;
    if((pthread_create(&tid,NULL,launch_kernel_thread,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }
    
    if((pthread_create(&tid,NULL,dispatcher,NULL))!=0){
        cout<<"can't create thread\n"<<endl;
    }

   vector<pthread_t> tid_arr;
   for(int i=0;i< PROCESS_THREAD_NUM;i++){
       if((pthread_create(&tid,NULL,processer,(void *)i))!=0){
           cout<<"can't create thread\n"<<endl;
       }else{
           tid_arr.push_back(tid);
       }
   }

   
   
   gettimeofday(&start, NULL); 
   for(int i= 0;i<PROCESS_THREAD_NUM;i++){
       pthread_join(tid_arr[i],NULL);
   }
   gettimeofday(&end, NULL);
   total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;  

    cout<<"multigpu prefix sort 4thread"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<ngpus * total_per_gpu * Para_Search_Bucket<<endl;
    for (int i=0;i<ngpus;i++) {
        cudaSetDevice(i+GPU_START);
        cudaEventSynchronize(g_stop[i]);
          
        cudaEventElapsedTime(&time_gpu,g_start[i],g_stop[i]);
        cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    }
    
    cout<<"total_time:              "<<total_time<<endl;
    //compute time 
    /*
    double tmp = 0;
    for(auto t:cpu_thread_compute_time){
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
   */ 
    for (int i=0;i<ngpus;i++) {
        
        cudaSetDevice(i+GPU_START);
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



