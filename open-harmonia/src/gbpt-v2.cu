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
#define Thread_Pre_Search 8
#define M 1000000
#define Keys_Count  100*M
#define Block_Dim 128
#define Block_Pre_SM 1024


#define LOAD_BALANCE_LEVEL 0

typedef struct {
    BLeaf_node *blfnode;
    int relist_idx;
}GPU_Result;

using namespace std;

namespace{
/*{{{*/
    //volatile  int stream_flag[2] = {-1,-1};
    volatile int **stream_flag;

    int Thread_Pre_Block = Block_Dim;
    int Block_Pre_Grid = SM * Block_Pre_SM;
    const int Para_Search_Bucket = Block_Pre_Grid * Thread_Pre_Block / Thread_Pre_Search;
    
    
    //cudaStream_t stream[2];
    cudaStream_t **stream;
    
    dim3 block_dim(Thread_Pre_Block);
    dim3 grid_dim(Block_Pre_Grid);

    //size:
    int d_keys_size = sizeof(key_t)*Para_Search_Bucket;
    int batch_gresult_size = sizeof(GPU_Result)*Para_Search_Bucket;
    int d_cur_inner_node_size = sizeof(Inner_node *)*Para_Search_Bucket;

    //tree:
    int rootIdx;
    int tree_height;
    int level = LOAD_BALANCE_LEVEL;
    Inner_node *root;
    Inner_node *inner_pool_start;
    
    //host: 
    //key_t *host_keys;
    //GPU_Result *h_gresult ;
    //Inner_node **cur_inner_node;
    
    
    //host:
    key_t **host_keys;
    GPU_Result **h_gresult;
    Inner_node ***cur_inner_node;
   


    //device:
    //Inner_node *d_innode;
    //key_t * d_keys;
    //GPU_Result *d_gresult;
    //Inner_node **d_cur_inner_node;



    //device:
    Inner_node **d_innode;
    key_t ** d_keys;
    GPU_Result **d_gresult;
    Inner_node ***d_cur_inner_node;


    const int total = Keys_Count / Para_Search_Bucket ;
    int total_per_gpu;

    void CUDART_CB CallBack(cudaStream_t stream, cudaError_t status, void *data) {
        stream_flag[(size_t)data/total_per_gpu][(size_t)data%total_per_gpu%2] = (size_t)data; 
    }



    cudaEvent_t g_start,g_stop;
    //volatile int key_status = -1;
    volatile int* key_status;

    int ngpus;
    vector<string> time_names;
    vector<float> times;
/*}}}*/
}

static __global__ void bpt_search_kernel_balance(Inner_node *d_innode,Inner_node ** d_cur_level_node/*当前搜索到的node*/,Inner_node *h_innode/*用于计算innernode idx*/,key_t *d_keys,int tree_height,GPU_Result *d_gresult,int key_count){    
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



static void * load_balance_CPU(void *  arg){// similiar to search_swp
    assert(level<tree_height);/*{{{*/
    struct timeval start;
    struct timeval end;

    int idx = 0;
    const int count = Para_Search_Bucket /P;    //P in bpt.h is 16
    gettimeofday(&start, NULL);



    while(idx<total_per_gpu){

        int start = idx*Para_Search_Bucket;
        
        for (int iii = 0; iii< ngpus;iii++) {
#ifdef CPU_8_CORE 
    omp_set_num_threads(4);
#else 
    omp_set_num_threads(12);//56 core
#endif
            #pragma omp parallel for  
            for(int i=0;i<count;i++){
                vector<Inner_node *> nodes(P,root);
                int relist_idx[P];
                for(int step = 1;step <=level;step++){
                    for(int k = 0;k<P;k++){
                        relist_idx[k] = nodes[k]->getChildIdx_avx2(NULL,host_keys[iii][start+i*P+k]);
                        nodes[k] = static_cast<Inner_node *>(((Inner_node *)nodes[k])->child[relist_idx[k]]);
                        __builtin_prefetch(nodes[k],0,3);
                    }
                }
                for(int k=0;k<P;k++) cur_inner_node[iii][start+i*P+k] = nodes[k];
            }

            key_status[iii]++;
        }
        
        idx++;
    }
    gettimeofday(&end, NULL);
    times[0] = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    //cout<<"CPU load balance"<<(end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0<<endl;
    return NULL;/*}}}*/
}


static void* launch_kernel_balance(void *args) {
      /*{{{*/
               cudaEvent_t *start;
        cudaEvent_t *stop;
        start = (cudaEvent_t*)malloc(ngpus * sizeof(cudaEvent_t));
        stop = (cudaEvent_t*)malloc(ngpus * sizeof(cudaEvent_t));

       //change-yzf 
	if(level==0){
		for(int iii=0;iii<ngpus;iii++){
			while(key_status[iii]<total_per_gpu-1){
				//printf("%d\n",key_status[iii]);
			};
		}
	}
        for (int i=0;i<ngpus;i++) {
            cudaSetDevice(i+GPU_START);
            cudaEventCreate(&(start[i]));
            cudaEventCreate(&(stop[i]));
            cudaEventRecord(start[i]);
        }

        for(int i=0;i<total_per_gpu;i++){

            int idx = i%2;
            for (int iii=0;iii<ngpus;iii++) {
                cudaSetDevice(iii+GPU_START);
                if(level!=0)while(key_status[iii]<i);//wait for load balance cpu 
            
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_cur_inner_node[iii] + idx*Para_Search_Bucket,  cur_inner_node[iii] + i*Para_Search_Bucket, d_cur_inner_node_size,cudaMemcpyHostToDevice,stream[iii][idx]));
            
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(d_keys[iii] + idx*Para_Search_Bucket,  host_keys[iii] + i*Para_Search_Bucket, d_keys_size,cudaMemcpyHostToDevice,stream[iii][idx]));
                
                bpt_search_kernel_balance<<<grid_dim,block_dim,0,stream[iii][idx]>>>(d_innode[iii],d_cur_inner_node[iii] + idx*Para_Search_Bucket,inner_pool_start,d_keys[iii]+idx*Para_Search_Bucket,tree_height-level,d_gresult[iii]+idx*Para_Search_Bucket,Para_Search_Bucket);
                
                CUDA_ERROR_HANDLER(cudaMemcpyAsync(h_gresult[iii] + i*Para_Search_Bucket,  d_gresult[iii] + idx*Para_Search_Bucket, batch_gresult_size, cudaMemcpyDeviceToHost, stream[iii][idx]));
                
                cudaStreamAddCallback(stream[iii][idx], CallBack, (void *)(i+iii*total_per_gpu), 0);
            }


        }
        for (int i=0;i<ngpus;i++) {
            cudaSetDevice(i+GPU_START);
            cudaEventRecord(stop[i]);
            cudaEventSynchronize(stop[i]);
            float time = 0;
            cudaEventElapsedTime(&time,start[i],stop[i]);
            times[1+i] = time/1000.0;
//cout<<"GPU time  "<<time/1000.0<<endl;
        }
        return NULL;
    /*}}}*/
}





void BPT_Search_GPU_multi_gpu_v4(BPlusTree &tree,ifstream &search_file){
/*{{{*/
    //cudaProfilerStart();
         time_names.push_back("CPU load balance");
        time_names.push_back("GPU");
        time_names.push_back("GPU");
        time_names.push_back("CPU load balance down");
        time_names.push_back("total");
        times.resize(5,0);

    //cudaGetDeviceCount(&ngpus);
    ngpus = GPU_NUM;
    d_innode = (Inner_node**)malloc(ngpus * sizeof(Inner_node*));

    for (int i=0;i<ngpus;i++) {
        CUDA_ERROR_HANDLER(cudaSetDevice(i+GPU_START));
        d_innode[i] = prepareGPU(tree);
    }

    rootIdx = tree.getRootIdx();
    root = static_cast<Inner_node *>(tree.getRoot());

    unsigned int tmp_size;
    tree.getInnerSegementationInfo(inner_pool_start,tmp_size);

    tree_height = tree.getHeight();
    assert(rootIdx != -1);


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
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_keys[iii]),d_keys_size *2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_gresult[iii]),batch_gresult_size * 2));
        CUDA_ERROR_HANDLER(cudaMalloc(&(d_cur_inner_node[iii]), d_cur_inner_node_size*2));
        
    }
    
    struct timeval start;
    struct timeval end;
    struct timeval start1;
    struct timeval end1;
    double total_time=0;
    
    
    gettimeofday(&start, NULL); 
   
    key_status = (volatile int *)malloc(ngpus * sizeof(int));
    for (int i = 0;i<ngpus;i++) key_status[i] = -1;

    
    
    //load_balance_CPU(NULL);
    pthread_t ntid;
    if ((pthread_create(&ntid,NULL, load_balance_CPU, NULL))!=0) {
        printf("can't create thread!\n");
    }
    
    stream_flag = (volatile int**)malloc(ngpus * sizeof(int*));
    for (int i=0;i<ngpus;i++) {
        stream_flag[i] = (volatile int*)malloc(2*sizeof(int));
        stream_flag[i][0] = -1;
        stream_flag[i][1] = -1;
    }
    
    pthread_t ntid1;
    if ((pthread_create(&ntid1, NULL, launch_kernel_balance, NULL))!=0) {
        printf("can't create thread!\n");
    }
    value_t val[Para_Search_Bucket]; 
    double tmp_time = 0;
    
    for (int idx=0; idx<total_per_gpu;idx++){
        int i = idx%2;
        //do cpu
        int start = idx*Para_Search_Bucket;

        for (int iii=0;iii<ngpus;iii++) {
        
            while(stream_flag[iii][i]<idx) ;
	    //cout<<"here"<<endl;
            gettimeofday(&start1, NULL); 
        
            if(key_status[iii]<total_per_gpu-1){
#ifdef CPU_8_CORE 
    omp_set_num_threads(10);
#else 
    omp_set_num_threads(40);//56 core
#endif

            }
        
            #pragma omp parallel for  
            for(int j=0;j<Para_Search_Bucket;j++){
                key_t key = host_keys[iii][start+j];
                volatile GPU_Result **new_ptr = (volatile GPU_Result **)h_gresult;
                BLeaf_node *blfnode =  h_gresult[iii][start+j].blfnode;
                while (blfnode==NULL) {
                    //cout<<key<<endl;
                    //cout<<"iii:"<<iii<<"j"<<j<<"wrong!!!!"<<endl;
                    //cout<<h_gresult<<endl;
                    //cout<<key<<endl;
                    //continue;
                    blfnode = new_ptr[iii][start+j].blfnode;
                }
                val[j] = blfnode->findKey(h_gresult[iii][start+j].relist_idx,key);
                //if(val[j] == -1)continue;
                //cout<<key<<": "<<val[j]<<endl;
                //if(key!= val1)     cout<<key<<": "<<val1<<endl;
            }  
            gettimeofday(&end1, NULL); 
            tmp_time += (end1.tv_sec - start1.tv_sec) + (end1.tv_usec-start1.tv_usec) / 1000000.0;
        
            //test
            
            //for (int j=0;j<Para_Search_Bucket;j++) {
            //    key_t key = host_keys[iii][start+j];
            //    cout<<key<<": "<<val[j]<<endl;
            //}
            


        }



    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0;
    cout<<"HB+ tree [double buffer load balance]"<<endl;
    cout<<"Search count "<<total*Para_Search_Bucket<<endl;
    cout<<"total bucket num "<<total<<endl;
    times[3] = tmp_time;
    times[4] = total_time;
    //cout<<"CPU time "<<tmp_time<<endl;
    //cout<<"GPU double_buffering total time:          "<<total_time<<endl;;
        // search GPU for rest of key 
    for(int i=0;i<5;i++){
        cout<< time_names[i]<<"time :"<<times[i]<<endl;
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

