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

#define M 1000000
#define Keys_Count 100*M

#define  GPU_SWP_SIZE1 (16/2) 


#define PPI_Block_Dim_1thread 128
#define PPI_Block_Per_SM 64

#define PPI_Thread_Per_Query_1thread 1
#define  Sort_Per_1Thread (GPU_SWP_SIZE1 / PPI_Thread_Per_Query_1thread)

#define CPU_THREAD 54 


#define LOAD_BALANCE_LEVEL 1 //CPU maybe compute last N level inner_node
#define LOAD_BALANCE_PECENTAGE 0 //unit %; Range 0~1 ,0 means CPU compute 0% inner_node,100% means CPU compute compute all inner_node

//thread scheduling 
#define PROCESS_CHRUNK_SIZE 1024  
#define DISPATCHER_THREAD_NUM 1

#define thread_52
#ifdef thread_52
#define PROCESS_THREAD_NUM 52
#else 
#define PROCESS_THREAD_NUM 26
#endif

//#define SPLIT_BUFFER_NUM (PROCESS_THREAD_NUM / 4)
#define SPLIT_BUFFER_NUM 2


//#define ENABLE_TEST 




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
    volatile int stream_flag[2] = {-1,-1};/*{{{*/
    int Thread_Per_Block = PPI_Block_Dim_1thread;
    int Block_Per_Grid = 13 * PPI_Block_Per_SM ;
    int Para_Search_Bucket = (Block_Per_Grid * Thread_Per_Block) / PPI_Thread_Per_Query_1thread * GPU_SWP_SIZE1;//851968

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
   
    //load balance 
    int bucket_balance_keys = ((int)((Para_Search_Bucket * LOAD_BALANCE_PECENTAGE+1024-1)/ 1024)) * 1024;//1024：the num of key which Theadblock process

    //thread scheduling
    key_t *range;
    vector<LBBuffer> buffers(SPLIT_BUFFER_NUM);
    vector<Thread_To_LBBuffer> thread_buffer_mapping(PROCESS_THREAD_NUM);


    unsigned long long  compare_times[2] = {0,0};   //compare_times[0]: only_index;     compare_times[1]: only_key
    unsigned long long *d_compare_times;
/*}}}*/
}

//---------------------------------------------------------------

static void stick_this_thread_to_core2(int core_id){
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);/*{{{*/
   
   //core_id*=2;
    //core_id = core_id % num_cores;
#ifdef thread_52
    static const int arr[] = {0,28,2,30,4,32,6,34,8,36,10,38,12,40,14,42,16,44,18,46,20,48,22,50,24,52,26,54,1,29,3,31,5,33,7,35,9,37,11,39,13,41,15,43,17,45,19,47,21,49,23,51,25,53,27,55};
#else
    static const int arr[] = {0,28,2,30,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25,27};
#endif    
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


__global__ void compare_times_1thread_kernel(Inner_node *d_innode, int* d_prefix, int inner_node_size_wo_last_inner_node, int root_idx,  key_t *d_keys, int tree_height, GPU_Result *d_gresult, int key_count,int d_bucket_balance_keys, unsigned long long* compare_times){
    /*{{{*/ 

    int key_base = PPI_Block_Dim_1thread / PPI_Thread_Per_Query_1thread * GPU_SWP_SIZE1;
    int key_idx = key_base * blockIdx.x + threadIdx.x/ PPI_Thread_Per_Query_1thread;  

    if(key_idx>=key_count) return;

    int search_idx = threadIdx.x % PPI_Thread_Per_Query_1thread;
    int query_idx_in_local_block = threadIdx.x / PPI_Thread_Per_Query_1thread;
    const int r = query_idx_in_local_block;//just for simple

    const int row = PPI_Block_Dim_1thread/ PPI_Thread_Per_Query_1thread;//blockDim.x/ PPI_Thread_Per_Query;
    const int row_swp = row * GPU_SWP_SIZE1;
    __shared__ char flag[row][3];
    __shared__ int inner_index_result[row];
    __shared__ int start_idx[row_swp];

    __shared__ char nexthalf[row];

    int stride = PPI_Block_Dim_1thread / PPI_Thread_Per_Query_1thread;
    
    /*
    for(int i=0;i<GPU_SWP_SIZE1;i++){
        int cur_key = key_idx+stride*i;
        if(cur_key>=key_count)continue;
        start_idx[r + stride*i] = d_inter_result[cur_key];
    }
*/
 
    for (int k = 0; k<Sort_Per_1Thread; k++){
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
        for(int j=0;j<GPU_SWP_SIZE1;j++){
            // get key
            __syncthreads();

            int cur_key = key_idx+stride *j;
            if(cur_key>=key_count)continue;
            int cur_r = r+stride *j;

            target = d_keys[cur_key];
            GPU_Result &result=d_gresult[cur_key];

            int pos = start_idx[cur_r];
            Inner_node *node = d_innode + pos;
            //search index;
            //key_t idx_key = node->inner_index[search_idx];
            int idx = 0;
            for(;idx<8;idx++){
                if(node->inner_index[idx] > target) break;
            }
           
            atomicAdd(&(compare_times[0]), idx+1);

            int begin = idx*8;

            idx = begin;
            for(;idx<begin+8;idx++){
                if(target < node->inner_key[idx]) break;
            }

            atomicAdd(&(compare_times[1]), idx-begin+1);
            

            if(i==tree_height-1){
                if(isBalance){
                        result.blfnode = __ldg(&d_prefix[pos])+idx ;
                        result.relist_idx = 111;
                }else{ 
                        result.blfnode = pos - inner_node_size_wo_last_inner_node;//useless
                        result.relist_idx = idx;
               }    

            }else 
                start_idx[cur_r] = __ldg(&d_prefix[pos]) + idx;
            __syncthreads();

        }
    }/*}}}*/
}




/*--------------------------------------------------------------------------
*
*   
*           PPI_BPT_V9
*           double buffer
*           using 2 thread
*           batch
*           key sort first, pos back
*           whole
*           new tree 
*           CPU multi-thread
*           load-balance ;GPU up,cpu down;
*           *thread_scheduling
*
*-----------------------------------------------------------------------------*/

static void* launch_kernel_thread(void *args){
/*{{{*/
    stick_this_thread_to_core2(1);
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

        compare_times_1thread_kernel<<<grid_dim, block_dim,0,stream[idx]>>>(d_innode,d_prefix, inner_node_size_wo_last_inner_node, rootIdx, d_keys_after+stride, kernel_height , d_gresult+stride, Para_Search_Bucket, bucket_balance_keys, d_compare_times);
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult+i*Para_Search_Bucket, d_gresult+stride, batch_gresult_size, cudaMemcpyDeviceToHost,stream[idx]));
        CUDA_ERROR_HANDLER(cudaMemcpyAsync(host_pos+i*Para_Search_Bucket, d_pos_after+stride, batch_pos_size, cudaMemcpyDeviceToHost,stream[idx]));

        cudaStreamAddCallback(stream[idx],CallBack,(void *)i,0);
    }
    cudaEventRecord(g_stop);
    return NULL;/*}}}*/
}

static void * dispatcher(void *args){
 /*{{{*/
    stick_this_thread_to_core2(2);

    vector<BufferWorkLeft> workLeftVector(SPLIT_BUFFER_NUM) ;

    for(int i=0;i<total;i++){
        int idx = i%2;
        while(stream_flag[idx] < i);
        int start = i * Para_Search_Bucket;

        //dispatcher
        int cur_idx = 0;

        
        for(int j=0;j<Para_Search_Bucket;j+= PROCESS_CHRUNK_SIZE ){
            int idx= start+j;
            key_t key = host_keys[start + host_pos[idx]];
            if(cur_idx < SPLIT_BUFFER_NUM-1 && key >= range[cur_idx]){
                cur_idx++;
            }
            buffers[cur_idx].put(idx,idx + PROCESS_CHRUNK_SIZE - 1,i);
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
        int bucket_idx= get<2>(range);


        int bucket_base = bucket_idx * Para_Search_Bucket;
        
        if(start == -1) {
            return NULL;
        }

        for(int k = start,j=0;k <=end;k++,j++){ 

            key_t key = host_keys[bucket_base + host_pos[k]];
            
            if(h_gresult[k].relist_idx == 111){ 
                
                void *node = (void *)(bptree->getInnerNodeByIdx(h_gresult[k].blfnode));

                int relist_idx = 0;
                
                for(int i=1;i<=LOAD_BALANCE_LEVEL;i++){
                    relist_idx = ((Inner_node *)node)->getChildIdx_avx2(NULL,key);
                    node = ((Inner_node *)node)->child[relist_idx];
                    //__builtin_prefetch(node,0,3);
                }
                key_t key = host_keys[bucket_base + host_pos[k]];
                vals[j]=((BLeaf_node *)node)->findKey(relist_idx,key);
            }else{
                BLeaf_node *blfnode = bptree->getLeafByIdx(h_gresult[k].blfnode);
                vals[j] = blfnode->findKey(h_gresult[k].relist_idx,key);
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
            key_t key = host_keys[bucket_base + host_pos[k]];
            //if(vals[j]!=key)
                //cout<<threadID<<" "<<key<<" "<<vals[j]<<endl;
            printf("%d %lld %lld\n", threadID, key, vals[j]);
        }
#endif
        
    }
/*}}}*/
}

void compare_times_1thread(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    stick_this_thread_to_core2(0);
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    
    
    bptree = &tree;
    prepareGPU_v2(tree, d_innode, d_prefix);

//for thread scheduling 
    range = (key_t *)malloc((SPLIT_BUFFER_NUM-1)*sizeof(key_t));
    tree.getKeyRangeOfInnerNode(tree.getHeight()-1-LOAD_BALANCE_LEVEL, SPLIT_BUFFER_NUM, range);

    for(int i=0;i<PROCESS_THREAD_NUM;i++){

        //thread_buffer_mapping[i].setMapping(i/4);//init 4 thread per buffer;
        //buffers[i/4].pushThread(i);
#ifdef thread_52
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
    
    //cout<<bucket_balance_keys<<endl;
    //cout<<Para_Search_Bucket<<endl;
    //cout<<nums<<endl;


    CUDA_ERROR_HANDLER(cudaMallocHost(&h_gresult,sizeof(GPU_Result)*nums));
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_pos,sizeof(int)*nums));

    //init key pos
    for (int i=0;i<Para_Search_Bucket;i++) {
        host_pos[i] = i;
        host_pos[i+Para_Search_Bucket] = i;
    }

    for(int i=0;i<2;i++) cudaStreamCreate(&stream[i]);


    //compare_collect
    CUDA_ERROR_HANDLER(cudaMalloc(&d_compare_times, sizeof(unsigned long long)*2));
    CUDA_ERROR_HANDLER(cudaMemcpy(d_compare_times, compare_times, sizeof(unsigned long long )*2,cudaMemcpyHostToDevice));



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

    cudaEventSynchronize(g_stop);
    cudaEventElapsedTime(&time_gpu,g_start,g_stop);
    cout<<"collect compare times"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU search total num:"<<total * Para_Search_Bucket<<endl;
    cout<<"GPU time(one stream):     "<<time_gpu/1000<<endl;
    cout<<"total_time:              "<<total_time<<endl;

    
    
    CUDA_ERROR_HANDLER(cudaMemcpy(compare_times, d_compare_times, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cout<<"1thread idx compare times        "<<(unsigned long long)compare_times[0]<<endl;
    cout<<"1thread idx&key compare times    "<<(unsigned long long)compare_times[0]+compare_times[1]<<endl;
    cout<<"8thread idx compare times        "<<(unsigned long long)total * Para_Search_Bucket* 8 * (kernel_height-1) <<endl;
    cout<<"8thread idx&key compare times    "<<(unsigned long long)total * Para_Search_Bucket* 16 * (kernel_height-1) <<endl;
    
    
    
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



