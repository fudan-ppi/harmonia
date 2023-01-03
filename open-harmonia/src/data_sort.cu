#include<cuda_runtime.h>
#include<iostream>
#include<fstream>
#include"../cub/cub/cub.cuh"
#include"cuda_utils.h" 
#include<sys/time.h>


#define key_t long long



using namespace std;


int main(int argc, char *argv[]) {
    if (argc<=1)return 0; 
    
    struct timeval start,end;

    ifstream search_file;
    search_file.open(argv[1]);

    int b_bit = 0;
    int e_bit = 64;

    b_bit = atoi(argv[2]);
    
    int Para_Search_Bucket = 13 * 64 * 128 / 2*16;
    
    key_t *host_keys; 
    CUDA_ERROR_HANDLER(cudaMallocHost(&host_keys,sizeof(key_t)*Para_Search_Bucket));
   
    key_t *d_keys;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys, sizeof(key_t)*Para_Search_Bucket));

    key_t *d_keys_out;
    CUDA_ERROR_HANDLER(cudaMalloc(&d_keys_out, sizeof(key_t)*Para_Search_Bucket));


    //read the key
    int nums = 0;
    string s;
    while((nums<Para_Search_Bucket)&&(getline(search_file,s))){
        key_t key;
        sscanf(s.c_str(),"%lld",&key);
        host_keys[nums++] = key;
    }

   
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CUDA_ERROR_HANDLER(cudaMemcpy(d_keys, host_keys, sizeof(key_t)*Para_Search_Bucket,cudaMemcpyHostToDevice));

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, d_keys_out, Para_Search_Bucket);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    gettimeofday(&start,NULL);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, d_keys_out, Para_Search_Bucket,b_bit, e_bit);
    cudaDeviceSynchronize();
    gettimeofday(&end,NULL);

    CUDA_ERROR_HANDLER(cudaMemcpy(host_keys, d_keys_out, sizeof(key_t)*Para_Search_Bucket,cudaMemcpyDeviceToHost));

    cout<<"sort time:"<< end.tv_sec-start.tv_sec + (end.tv_usec-start.tv_usec)/1000000.0<<endl;

    for (int i=0;i<Para_Search_Bucket;i++)
        cout<<host_keys[i]<<endl;
   

}
