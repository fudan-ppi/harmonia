//#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include "cpu-search.h"
#include "conf.h"
#include <papi.h>

#ifdef BOOST

#include <boost/sort/spreadsort/spreadsort.hpp>
#define BUCKET_SIZE 100000
#define SWP_P   16          //same use with P in bpt.h
#ifdef PAPI_PROFILE
#ifndef PAPI_VAR
#define PAPI_VAR
long long l1dmiss=0;
long long l2dmiss=0;
long long l2daccess=0;
long long ldIns=0;
long long brcn=0;
long long brtkn=0;
int EventSet1=PAPI_NULL;
          
        const int numEvents=3;
        long long papiS[numEvents]={0};
        long long papiE[numEvents]={0};
#endif
#endif
// sort 
// prefetch 
//
using namespace std;
using namespace boost::sort::spreadsort;
void search_cpu_prefetch_sort_NTG(ifstream &file, BPlusTree &tree){
/*{{{*/
    struct timeval start;
    struct timeval end;
    string s;
    key_t key;
    vector<key_t> keys;
   
#ifdef PROFILING 
    long long cmp_times_fact = 0;
    long long  cmp_times_ideal = 0;
    long long tmp_cmp_fact = 0;
    long long tmp_cmp_ideal = 0;
#endif 
    
  
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
    
    
#ifdef PAPI_PROFILE 
    /*{{{*/
    if(PAPI_library_init(PAPI_VER_CURRENT)!=PAPI_VER_CURRENT){
        cout<<"error library init"<<endl;
        exit(-1);
    }
    if(PAPI_thread_init((unsigned long(*)(void))(omp_get_thread_num))!=PAPI_OK){
        cout<<"error thread init"<<endl;
        exit(1);
    }
    int retval=PAPI_create_eventset(&EventSet1);
    if(retval !=PAPI_OK){
        cout<<"error when create eventset."<<endl;
        exit(-1);
    }
    
    retval=PAPI_add_event(EventSet1,PAPI_L1_DCM);
    if(retval!=PAPI_OK){
            cout<<"error when add event BR CN."<<endl;
            exit(-1);
    } 

    retval=PAPI_add_event(EventSet1,PAPI_L2_DCM);
    if(retval!=PAPI_OK){
            cout<<"error when add event BR CN."<<endl;
            exit(-1);
    } 
/*}}}*/
#endif


 
    gettimeofday(&start, NULL);
    int p_size = keys_size / SWP_P;  
    if (tree.isPPItree){
#ifdef PAPI_PROFILE
/*{{{*/
        if(PAPI_start(EventSet1)!=PAPI_OK)
            exit(-1);
        if(PAPI_read(EventSet1,papiS)!=PAPI_OK){
           exit(-1); 
        }
/*}}}*/
#endif 

#if defined(PAPI_PROFILE) || defined(PROFILING) 
#else 
        #pragma omp parallel for
#endif
        for (int k=0;k<p_size;k++){   
#ifdef PROFILING 
            search_cpu_prefetch_NTG_PPI_profile(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree, tmp_cmp_fact, tmp_cmp_ideal);
            cmp_times_ideal += tmp_cmp_ideal;
            cmp_times_fact += tmp_cmp_fact;
#else 
            search_cpu_prefetch_NTG_PPI(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
#endif  
        }

#ifdef PAPI_PROFILE   
/*{{{*/
        if(PAPI_stop(EventSet1,papiE)!=PAPI_OK){
                cout<<"papi stop error."<<endl;
                exit(-1);
        }
     //       ldIns=papiE[0]-papiS[0];
        l1dmiss=papiE[0]-papiS[0];
        l2dmiss=papiE[1]-papiS[1];
        if(PAPI_cleanup_eventset(EventSet1)!=PAPI_OK){
                cout<<"papi cleanup error."<<endl;
                exit(-1);
        }
        if(PAPI_destroy_eventset(&EventSet1)!=PAPI_OK){
                cout<<"destroy eventset error"<<endl;
                exit(-1);
        }
       
        PAPI_shutdown();
/*}}}*/
#endif
    }else{

        #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_NTG_HB(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
 

    }

    
    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: cpu prefetch sort avxNTG  "<<endl;
    else cout<<"HB+ [no gpu]: cpu prefetch sort avxNTG"<<endl;

#ifdef PROFILING 
    cout<<"cmp_times_ideal:     "<<cmp_times_ideal<<endl;
    cout<<"cmp_times_fact:     "<<cmp_times_fact<<endl;
    cout<<"cmp_times_waste:     "<<cmp_times_fact - cmp_times_ideal<<endl;
#endif

    cout<<"Search_size: "<<keys_size<<endl;
    cout<<"Sort time:      "<<sort_time<<endl;
    cout<<"Search time:      "<<tt<<endl;
    cout<<"Total time:      "<<tt+sort_time<<endl;
#ifdef PAPI_PROFILE    
//  cout<< "Load Ins\t"<<ldIns<<endl;   
    cout<<"L1 miss\t"<<l1dmiss<<endl;
    cout<<"L2 miss\t"<<l2dmiss<<endl;
 /*   cout<<"BR_TKN\t"<<papi[1]<<"   "<<papiStart[1]<<"    "<<papiEnd[1]<<endl;
    cout<<"BR_NTK\t"<<papi[2]<<"   "<<papiStart[2]<<"    "<<papiEnd[2]<<endl;
 */
  
#endif
/*}}}*/

}



void search_cpu_prefetch_NTG(ifstream &file, BPlusTree &tree){

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
#ifdef PROFILING 
    long long cmp_times_fact = 0;
    long long  cmp_times_ideal = 0;
    long long tmp_cmp_fact = 0;
    long long tmp_cmp_ideal = 0;
#endif 
    
#ifdef PAPI_PROFILE 
    /*{{{*/
    if(PAPI_library_init(PAPI_VER_CURRENT)!=PAPI_VER_CURRENT){
        cout<<"error library init"<<endl;
        exit(-1);
    }
    if(PAPI_thread_init((unsigned long(*)(void))(omp_get_thread_num))!=PAPI_OK){
        cout<<"error thread init"<<endl;
        exit(1);
    }
    int retval=PAPI_create_eventset(&EventSet1);
    if(retval !=PAPI_OK){
        cout<<"error when create eventset."<<endl;
        exit(-1);
    }
    
    retval=PAPI_add_event(EventSet1,PAPI_LD_INS);
    if(retval!=PAPI_OK){
            cout<<"error when add event BR CN."<<endl;
            exit(-1);
    } 

    retval=PAPI_add_event(EventSet1,PAPI_L2_DCA);
    if(retval!=PAPI_OK){
            cout<<"error when add event BR CN."<<endl;
            exit(-1);
    } 

    retval=PAPI_add_event(EventSet1,PAPI_L3_DCA);
    if(retval!=PAPI_OK){
            cout<<"error when add event BR CN."<<endl;
            exit(-1);
    } 


/*}}}*/
#endif


    gettimeofday(&start, NULL);
    int p_size = keys_size / SWP_P;  
    if (tree.isPPItree){
#ifdef PAPI_PROFILE
/*{{{*/
        if(PAPI_start(EventSet1)!=PAPI_OK)
            exit(-1);
        if(PAPI_read(EventSet1,papiS)!=PAPI_OK){
           exit(-1); 
        }
/*}}}*/
#endif 
#if defined(PAPI_PROFILE) || defined(PROFILING) 
#else 
        #pragma omp parallel for
#endif
        for (int k=0;k<p_size;k++){   
   
#ifdef PROFILING 
            search_cpu_prefetch_NTG_PPI_profile(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree, tmp_cmp_fact, tmp_cmp_ideal);
            cmp_times_ideal += tmp_cmp_ideal;
            cmp_times_fact += tmp_cmp_fact;
#else 
            search_cpu_prefetch_NTG_PPI(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
#endif
        }
#ifdef PAPI_PROFILE   
/*{{{*/
        if(PAPI_stop(EventSet1,papiE)!=PAPI_OK){
                cout<<"papi stop error."<<endl;
                exit(-1);
        }
           ldIns=papiE[0]-papiS[0];
           l2daccess=papiE[1]-papiS[1];
        l2dmiss=papiE[2]-papiS[2];
        if(PAPI_cleanup_eventset(EventSet1)!=PAPI_OK){
                cout<<"papi cleanup error."<<endl;
                exit(-1);
        }
        if(PAPI_destroy_eventset(&EventSet1)!=PAPI_OK){
                cout<<"destroy eventset error"<<endl;
                exit(-1);
        }
       
        PAPI_shutdown();
/*}}}*/
#endif
    }else{
#ifdef PAPI_PROFILE
/*{{{*/
        if(PAPI_start(EventSet1)!=PAPI_OK)
            exit(-1);
        if(PAPI_read(EventSet1,papiS)!=PAPI_OK){
           exit(-1); 
        }
/*}}}*/
#endif 

#if defined(PAPI_PROFILE) || defined(PROFILING) 
#else 
        #pragma omp parallel for
#endif
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_NTG_HB(&(keys[k*SWP_P]), &(values[k*SWP_P]), tree);
 
#ifdef PAPI_PROFILE   
/*{{{*/
        if(PAPI_stop(EventSet1,papiE)!=PAPI_OK){
                cout<<"papi stop error."<<endl;
                exit(-1);
        }
            ldIns=papiE[0]-papiS[0];
            l2daccess=papiE[1]-papiS[1];
            l2dmiss=papiE[2]-papiS[2];
        if(PAPI_cleanup_eventset(EventSet1)!=PAPI_OK){
                cout<<"papi cleanup error."<<endl;
                exit(-1);
        }
        if(PAPI_destroy_eventset(&EventSet1)!=PAPI_OK){
                cout<<"destroy eventset error"<<endl;
                exit(-1);
        }
       
        PAPI_shutdown();
/*}}}*/
#endif

    }

    
    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: cpu prefetch avxNTG  "<<endl;
    else cout<<"HB+ [no gpu]: cpu prefetch avxNTG"<<endl;
    cout<<"height:  "<<tree.getHeight()<<endl;
#ifdef PROFILING 
    cout<<"cmp_times_ideal:     "<<cmp_times_ideal<<endl;
    cout<<"cmp_times_fact:     "<<cmp_times_fact<<endl;
    cout<<"cmp_times_waste:     "<<cmp_times_fact - cmp_times_ideal<<endl;
#endif
    cout<<"Search_size: "<<keys_size<<endl;
    cout<<"Search time:      "<<tt<<endl;
    cout<<"Total time:      "<<tt<<endl;

#ifdef PAPI_PROFILE    
  cout<< "Load Ins\t"<<ldIns<<endl;   
    cout<<"L2 access\t"<<l2daccess<<endl;
    cout<<"L3 access\t"<<l2dmiss<<endl;
 /*   cout<<"BR_TKN\t"<<papi[1]<<"   "<<papiStart[1]<<"    "<<papiEnd[1]<<endl;
    cout<<"BR_NTK\t"<<papi[2]<<"   "<<papiStart[2]<<"    "<<papiEnd[2]<<endl;
 */
  
#endif
/*}}}*/

}


#ifdef TREE_32


void search_cpu_prefetch_NTG_HB(key_t* keys,value_t* vals,BPlusTree &tree){
/*{{{*/
    int height=tree.getHeight();
    void *root =tree.getRoot();
    
    void * nodes[SWP_P];
    for(int i=0;i<SWP_P;i++){
        nodes[i]=root;
    }
    int relist_idx[SWP_P];
    int SWP_P_div_8=SWP_P/8;
    for(int step=1;step<height;step++){
        for(int i=0;i<SWP_P_div_8;i++){
           tree.getChildIdx_avx_4keys((Inner_node**)&(nodes[i*8]),&(keys[i*8]),&(relist_idx[i*8])) ;
            for(int j=0;j<8;j++){
                nodes[i*8+j]=((Inner_node*) nodes[i*8+j])->child[relist_idx[i*8+j]];
                __builtin_prefetch(nodes[i*8+j],0,3);
            }
        }
    }
    for(int i=0;i<SWP_P;i++) vals[i]=((BLeaf_node*)nodes[i])->findKey(relist_idx[i],keys[i]);

/*}}}*/
}

void search_cpu_prefetch_NTG_PPI(key_t* keys,value_t* vals,BPlusTree &tree){
/*{{{*/
    int height=tree.getHeight();
    int * prefixArray=tree.getPrefixArray();
    int innerSize_wo_last_inner_node=tree.getInnerSize_wo_last_inner_node();
    int node_idx[SWP_P];
    for(int i=0;i<SWP_P;i++){
        node_idx[i]=0;
    }
    int relist_idx[SWP_P];
    Inner_node * nodes[SWP_P];
    int SWP_P_div_8=SWP_P/8;
    for(int step =1;step<height;step++){
        for(int i=0;i<SWP_P_div_8;i++){
            for(int j=0;j<8;j++){
                nodes[i*8+j]=tree.getInnerNodeByIdx(node_idx[i*8+j]);
                __builtin_prefetch(nodes[i*8+j],0,3);
            }
            tree.getChildIdx_avx_4keys((Inner_node**)&(nodes[i*8]), &(keys[i*8]), &(relist_idx[i*8]));
            for (int j=0;j<8;j++)
                node_idx[i*8+j]=prefixArray[node_idx[i*8+j]]+relist_idx[i*8+j];
        }

    }
    for (int i=0;i<SWP_P;i++){
        void * node=tree.getBLeafNodeByIdx(tree.getInnerNodeIdx(nodes[i])-innerSize_wo_last_inner_node);
        vals[i]=((BLeaf_node*)node)->findKey(relist_idx[i],keys[i]);
    }
    /*}}}*/
}

#else

void search_cpu_prefetch_NTG_HB(key_t* keys, value_t* vals, BPlusTree &tree){
/*{{{*/
    int height = tree.getHeight();
    void *root = tree.getRoot();
    
    void * nodes[SWP_P];
    for(int i=0;i<SWP_P;i++) {
        nodes[i] = root;
    }
    int relist_idx[SWP_P];
    
    int SWP_P_div_4 = SWP_P / 4;

    for(int step = 1; step<height; step++){
        for(int i = 0;i<SWP_P_div_4;i++){
            tree.getChildIdx_avx_4keys((Inner_node**)&(nodes[i*4]), &(keys[i*4]), &(relist_idx[i*4]));
            for (int j=0;j<4;j++) {
                nodes[i*4+j] = ((Inner_node *)nodes[i*4+j])->child[relist_idx[i*4+j]];
                __builtin_prefetch(nodes[i*4+j],0,3);
            }
        }
    }
    for(int i=0;i<SWP_P;i++) vals[i] = ((BLeaf_node*)nodes[i])->findKey(relist_idx[i],keys[i]);

   /*}}}*/
    
}


void search_cpu_prefetch_NTG_PPI(key_t* keys, value_t* vals, BPlusTree &tree){
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
    int SWP_P_div_4 = SWP_P / 4;
    
    for(int step = 1; step<height; step++){
        
 
          
        for(int i = 0;i<SWP_P_div_4;i++){
            for (int j=0;j<4;j++) {
                nodes[i*4+j] = tree.getInnerNodeByIdx(node_idx[i*4+j]);
                __builtin_prefetch(nodes[i*4+j],0,3);
                
            }
#ifdef PAPI_PROFILE
//        if(PAPI_read(EventSet1,papiS)!=PAPI_OK)
//        exit(-1);
#endif

            tree.getChildIdx_avx_4keys((Inner_node**)&(nodes[i*4]), &(keys[i*4]), &(relist_idx[i*4]));
    
            for (int j=0;j<4;j++)
                node_idx[i*4+j] = prefixArray[node_idx[i*4+j]] + relist_idx[i*4+j];   

#ifdef PAPI_PROFILE            
//    if(PAPI_read(EventSet1,papiE)!=PAPI_OK)
//        exit(-1);
//    brcn+=(papiE[0]-papiS[0]);
#endif
        }


    }

    for(int i=0;i<SWP_P;i++) {
        void *node = tree.getBLeafNodeByIdx(tree.getInnerNodeIdx(nodes[i])-innerSize_wo_last_inner_node);
        vals[i] = ((BLeaf_node*)node)->findKey(relist_idx[i],keys[i]);
    }
        

   /*}}}*/
    
}
#endif




#ifdef PROFILING


#ifdef TREE_32


void search_cpu_prefetch_NTG_HB_profile(key_t* keys,value_t* vals,BPlusTree &tree, long long &cmp_times_fact,long long &cmp_times_ideal){
}

void search_cpu_prefetch_NTG_PPI_profile(key_t* keys, value_t* vals, BPlusTree &tree, long long  &cmp_times_fact, long long &cmp_times_ideal){
}

#else

void search_cpu_prefetch_NTG_HB_profile(key_t* keys,value_t* vals,BPlusTree &tree, long long &cmp_times_fact,long long &cmp_times_ideal){
  
}


void search_cpu_prefetch_NTG_PPI_profile(key_t* keys, value_t* vals, BPlusTree &tree, long long &cmp_times_fact, long long &cmp_times_ideal){
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
    int SWP_P_div_4 = SWP_P / 4;
   
    int tmp_cmp_fact = 0; 
    int  tmp_cmp_ideal = 0;
    cmp_times_fact = 0;
    cmp_times_ideal = 0;
    for(int step = 1; step<height; step++){
        
 
          
        for(int i = 0;i<SWP_P_div_4;i++){
            for (int j=0;j<4;j++) {
                nodes[i*4+j] = tree.getInnerNodeByIdx(node_idx[i*4+j]);
                __builtin_prefetch(nodes[i*4+j],0,3);
                
            }


            tree.getChildIdx_avx_4keys_profile((Inner_node**)&(nodes[i*4]), &(keys[i*4]), &(relist_idx[i*4]), tmp_cmp_fact, tmp_cmp_ideal);
            cmp_times_fact += tmp_cmp_fact;
            cmp_times_ideal += tmp_cmp_ideal;

            for (int j=0;j<4;j++)
                node_idx[i*4+j] = prefixArray[node_idx[i*4+j]] + relist_idx[i*4+j];   


        }


    }

    for(int i=0;i<SWP_P;i++) {
        void *node = tree.getBLeafNodeByIdx(tree.getInnerNodeIdx(nodes[i])-innerSize_wo_last_inner_node);
        vals[i] = ((BLeaf_node*)node)->findKey(relist_idx[i],keys[i]);
    }
        

   /*}}}*/
    
}
#endif




#endif 




#endif
