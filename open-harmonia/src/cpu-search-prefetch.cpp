//#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include "cpu-search.h"
#include "conf.h"
#include <papi.h>

using namespace std;
#define BUCKET_SIZE 100000
#define SWP_P   16          //same use with P in bpt.h

#ifdef PAPI_PROFILE
#ifndef PAPI_VAR
#define PAPI_VAR
     const int numE=4;
    long long papiStart[numE]={0};
    long long papiEnd[numE]={0};
    long long papi[numE]={0};
    long long levelmiss={0};
    int EventSet=PAPI_NULL;
#endif
#endif

#ifdef BOOST

#include <boost/sort/spreadsort/spreadsort.hpp>
using namespace boost::sort::spreadsort;

void search_cpu_prefetch_sort(ifstream &file, BPlusTree &tree){
/*{{{*/
    struct timeval start;
    struct timeval end;
    string s;
    key_t key;
    vector<key_t> keys;
    
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        keys.push_back(key);
    }

    long keys_size = keys.size();
    int total = keys_size / BUCKET_SIZE; 
 
    gettimeofday(&start, NULL);
    #pragma omp parallel for
    for (int i=0;i<total;i++) {
        integer_sort(&(keys[i*BUCKET_SIZE]), &(keys[(i+1)*BUCKET_SIZE]), rightshift());
    }
    gettimeofday(&end, NULL);
    double sort_time =  (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;

    vector<value_t> values(keys_size);


    gettimeofday(&start, NULL);
    int p_size = keys_size / SWP_P;  
    if (tree.isPPItree){
        #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_PPI(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);

    }else{
        #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_HB(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
    }



 
    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: cpu prefetch sort "<<endl;
    else cout<<"HB+ [no gpu]: cpu prefetch sort"<<endl;
    cout<<"Search_size: "<<keys_size<<endl;
    cout<<"Sort time:      "<<sort_time<<endl;
    cout<<"Search time:      "<<tt<<endl;
    cout<<"Total time:      "<<tt+sort_time<<endl;

/*}}}*/

}

#endif

void search_cpu_prefetch(ifstream &file, BPlusTree &tree){
/*{{{*/
    struct timeval start;
    struct timeval end;
    string s;
    key_t key;
    vector<key_t> keys;
    
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        keys.push_back(key);
    }

    long keys_size = keys.size();
 
    vector<value_t> values(keys_size);
#ifdef PAPI_PROFILE
    if(PAPI_library_init(PAPI_VER_CURRENT)!=PAPI_VER_CURRENT){
        cout<<"error library init"<<endl;
        exit(-1);
    }
    if(PAPI_thread_init((unsigned long(*)(void))(omp_get_thread_num))!=PAPI_OK){
        cout<<"error thread init"<<endl;
        exit(1);
    }

    int retval=PAPI_create_eventset(&EventSet);
    if(retval !=PAPI_OK){
        cout<<"error when create eventset."<<endl;
    }
/* 
    retval=PAPI_add_event(EventSet,PAPI_L1_DCM);
    if(retval!=PAPI_OK){
        cout<<"error when add event BR CN."<<endl;
    } 
    retval=PAPI_add_event(EventSet,PAPI_L2_DCM);
    if(retval!=PAPI_OK){
        cout<<"error when add event L2 DCM;"<<endl;
    }
 */
/*
    retval=PAPI_add_event(EventSet,PAPI_BR_TKN);
    if(retval!=PAPI_OK){
        cout<<"error when add event L2 DCM;"<<endl;
    }
 
    retval=PAPI_add_event(EventSet,PAPI_BR_NTK);
    if(retval!=PAPI_OK){
        cout<<"error when add event L2 DCA;"<<endl;
    }

    */
    /*
    retval=PAPI_add_event(EventSet,PAPI_L1_DCM);
    if(retval!=PAPI_OK){
        cout<<"error when add event L1 DCM."<<endl;
    } 
*/
    /*
    retval=PAPI_add_event(EventSet,PAPI_L2_DCM);
    if(retval!=PAPI_OK){
        cout<<"error when add event L2 DCM;"<<endl;
    }
 */
    retval=PAPI_add_event(EventSet,PAPI_L2_DCA);
    if(retval!=PAPI_OK){
        cout<<"error when add event L2 DCA;"<<endl;
    }
 
    retval=PAPI_add_event(EventSet,PAPI_L3_TCM);
    if(retval!=PAPI_OK){
        cout<<"error when add event L3 DCA;"<<endl;
    }


 
    retval=PAPI_add_event(EventSet,PAPI_LD_INS);
    if(retval!=PAPI_OK){
        cout<<"error when add event LD INS;"<<endl;
    }
    retval=PAPI_add_event(EventSet,PAPI_L3_LDM);
    if(retval!=PAPI_OK){
        cout<<"error when add event L3 TCM;"<<endl;
    }
#endif
   
    gettimeofday(&start, NULL);
    int p_size = keys_size / SWP_P;  
    if (tree.isPPItree){
        #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_PPI(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
    }else{
#ifdef PAPI_PROFILE
        if(PAPI_start(EventSet)!=PAPI_OK)
            exit(-1);
        if(PAPI_read(EventSet,papiStart)!=PAPI_OK){
            cout<<"papi read error"<<endl;
            exit(-1);
        }
#endif

       // #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_HB(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);

#ifdef PAPI_PROFILE
        if(PAPI_stop(EventSet,papiEnd)!=PAPI_OK){
            cout<<"papi stop error."<<endl;
            exit(-1);
        }
        for(int i=0;i<numE;i++){
             papi[i]+=(papiEnd[i]-papiStart[i]);
        }
       if(PAPI_cleanup_eventset(EventSet)!=PAPI_OK){
            cout<<"papi cleanup error."<<endl;
            exit(-1);
        }
        if(PAPI_destroy_eventset(&EventSet)!=PAPI_OK){
            cout<<"destroy eventset error"<<endl;
            exit(-1);
        }
       
        PAPI_shutdown();
#endif

    }


    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: cpu prefetch  "<<endl;
    else cout<<"HB+ [no gpu]: cpu prefetch"<<endl;
    cout<<"Search_size: "<<keys_size<<endl;
    cout<<"Search time:      "<<tt<<endl;
#ifdef PAPI_PROFILE
     cout<<"Load Ins\t"<<papi[2]<<"   "<<papiStart[2]<<"    "<<papiEnd[2]<<endl;
 //   cout<<"L2 cache miss\t"<<papi[1]<<"   "<<papiStart[1]<<"    "<<papiEnd[1]<<endl;
     cout<<"L2 cache access\t"<<papi[0]<<"   "<<papiStart[0]<<"    "<<papiEnd[0]<<endl;
     cout<<"L3 TCM \t"<<papi[1]<<"   "<<papiStart[1]<<"    "<<papiEnd[1]<<endl;
     cout<<"L3 cache miss\t"<<papi[3]<<"   "<<papiStart[3]<<"    "<<papiEnd[3]<<endl;


 // cout<<"BR CN\t"<<papi[0]<<"   "<<papiStart[0]<<"    "<<papiEnd[0]<<endl;
  /*
    cout<<"BR_CN\t"<<papi[0]<<"   "<<papiStart[0]<<"    "<<papiEnd[0]<<endl;
    cout<<"BR_TKN\t"<<papi[1]<<"   "<<papiStart[1]<<"    "<<papiEnd[1]<<endl;
    cout<<"BR_NTK\t"<<papi[2]<<"   "<<papiStart[2]<<"    "<<papiEnd[2]<<endl;
  */
#endif
/*}}}*/

}



//avx function
void search_cpu_prefetch_HB(key_t* keys, value_t* vals, BPlusTree &tree){
    /*{{{*/
    int height = tree.getHeight();
    void *root = tree.getRoot();
    
    void * nodes[SWP_P];
    for(int i=0;i<SWP_P;i++) {
        nodes[i] = root;
    }
    int relist_idx[SWP_P];
    
    for(int step = 1; step<height; step++){
#ifdef PAPI_PROFILE 
    
#endif

        for(int i = 0;i<SWP_P;i++){
            relist_idx[i]=((Inner_node *)nodes[i])->getChildIdx(NULL,keys[i]);
#ifdef PAPI_PROFILE
// if(PAPI_read(EventSet,papiStart)!=PAPI_OK)
//            exit(-1);
#endif
            nodes[i] = ((Inner_node *)nodes[i])->child[relist_idx[i]];

#ifdef PAPI_PROFILE
//        if(PAPI_read(EventSet,papiEnd)!=PAPI_OK)
//            exit(-1);
//        papi[0]+=(papiEnd[0]-papiStart[0]);
#endif
        __builtin_prefetch(nodes[i],0,3);
        }

    }

    for(int i=0;i<SWP_P;i++) vals[i] = ((BLeaf_node*)nodes[i])->findKey(relist_idx[i],keys[i]);

   /*}}}*/
    
}


//avx function
void search_cpu_prefetch_PPI(key_t* keys, value_t* vals, BPlusTree &tree){
    /*{{{*/
    int height = tree.getHeight();
    int *prefixArray = tree.getPrefixArray();
    int innerSize_wo_last_inner_node = tree.getInnerSize_wo_last_inner_node();
    
    int node_idx[SWP_P]; //for PPItree only
    for(int i=0;i<SWP_P;i++) {
        node_idx[i] = 0;
    }
    
    int relist_idx[SWP_P];
    Inner_node * nodes[SWP_P];
    
    for(int step = 1; step<height; step++){
        for(int i = 0;i<SWP_P;i++){
            
            nodes[i] = tree.getInnerNodeByIdx(node_idx[i]);
            __builtin_prefetch(nodes[i],0,3);
            relist_idx[i]=((Inner_node *)nodes[i])->getChildIdx(NULL,keys[i]);
            node_idx[i] = prefixArray[node_idx[i]] + relist_idx[i];   
        }
    }
    for(int i=0;i<SWP_P;i++) {
        void *node = tree.getBLeafNodeByIdx(tree.getInnerNodeIdx(nodes[i])-innerSize_wo_last_inner_node);
        vals[i] = ((BLeaf_node*)node)->findKey(relist_idx[i],keys[i]);
    }
   
    /*}}}*/
}



