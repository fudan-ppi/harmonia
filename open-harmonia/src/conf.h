#ifndef CONF_H
#define CONF_H
#include<iostream>
#include<pthread.h>
#include<unistd.h>
#include "conf_for_bpt.h"

#define SM 80
#define GPU_NUM 1
#define GPU_START 0 // TITAN V change to 3

#define PPI_Block_Per_SM 512


//cpu core configure

//#define CPU_8_CORE 

#ifdef CPU_8_CORE 
#define CPU_PHYSICAL_CORE 8
#define CPU_LOGICAL_CORE 16
#else 
#define CPU_PHYSICAL_CORE 28
#define CPU_LOGICAL_CORE 56
#endif

#define OMP_Thread_num CPU_LOGICAL_CORE

#define thread_hyper
#ifdef thread_hyper 
//LOGICAL -4
    #ifdef CPU_8_CORE 
    #define PROCESS_THREAD_NUM 12 
    #else 
    #define PROCESS_THREAD_NUM 52 
    #endif 
#else 
//#define PROCESS_THREAD_NUM 26
#endif


//#define ENABLE_TEST

// No_transfer 's ENABLE_TEST is invaild for now
// 1. ONLY the last bucket'answer will be passed back!!
// 2. THE LAST LAYER ISN'T SEARCHED!

#define BOOST 
//#define PAPI_PROFILE
//#define PROFILING

//---------------------------------------------------------------

static void stick_this_thread_to_core2(int core_id){
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);/*{{{*/
   
   //core_id*=2;
    //core_id = core_id % num_cores;
#ifdef thread_hyper
    #ifdef CPU_8_CORE
        static const int arr[] = {0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23};
    #else 
        static const int arr[] = {0,28,2,30,4,32,6,34,8,36,10,38,12,40,14,42,16,44,18,46,20,48,22,50,24,52,26,54,1,29,3,31,5,33,7,35,9,37,11,39,13,41,15,43,17,45,19,47,21,49,23,51,25,53,27,55};
    #endif
#else
//static const int arr[] = {0,28,2,30,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25,27};
#endif    
    if(core_id<0 || core_id >= num_cores) {
        std::cout<<"core id wrong"<<std::endl;   
    }
    core_id = arr[core_id];
    cpu_set_t mask;
    //cpu_set_t get;
    CPU_ZERO(&mask);
    CPU_SET(core_id,&mask);
    if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)<0){
        std::cout<<"set thread affinity error"<<std::endl;
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

#endif
//#define CPU_THREAD (PROCESS_THREAD_NUM-2)
