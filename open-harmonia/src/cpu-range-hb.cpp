#define RANGE_HB
#ifdef RANGE_HB
//#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include "cpu-search.h"
#include "conf.h"


#define BUCKET_SIZE 100000      // parallel granularity for sort
#define SWP_P   16          //same use with P in bpt.h
                            //parallel granularity for prefetch search 
#define SWP_P2 1000           //parallel granularity for range traverse 
//PS: granularity for NTG avx is always 4(for 64bit) or 8(for 32bit)


#define Range_Size 8 

// sort 
// prefetch 
//
using namespace std;


void search_range_cpu_prefetch_HB(ifstream &file, BPlusTree &tree){
/*{{{*/
    struct timeval start;
    struct timeval end;
    string s;
    key_t key;
    vector<key_t> keys;
    vector<key_t> end_keys;
    
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        keys.push_back(key);
        end_keys.push_back(Max_Key);
    }

    long keys_size = keys.size();
    int total = keys_size / BUCKET_SIZE; 
    
    // HB no sort 
    
   // gettimeofday(&start, NULL);
   // #pragma omp parallel for
   // for (int i=0;i<total;i++) {
   //     integer_sort(&(keys[i*BUCKET_SIZE]), &(keys[(i+1)*BUCKET_SIZE]), rightshift());
   // }
   // gettimeofday(&end, NULL);
   // double sort_time =  (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;

    vector<BLeaf_node *> leafnodes(keys_size);
    vector<int> relist_idx(keys_size);
   
    gettimeofday(&start, NULL);
    int p_size = keys_size / SWP_P;  
    // for PPItree only for now 
    if (tree.isPPItree) 
        return;
        //#pragma omp parallel for
        //for (int k=0;k<p_size;k++)
        //    search_cpu_prefetch_NTG_PPI_wo_leaf(&(keys[k*SWP_P]),  &(leafnode_id[k*SWP_P]), &(relist_idx[k*SWP_P]), tree);
    else
        
        #pragma omp parallel for
        for (int k=0;k<p_size;k++) 
            search_cpu_prefetch_HB_wo_leaf(&(keys[k*SWP_P]), &(leafnodes[k*SWP_P]),&(relist_idx[k*SWP_P]) , tree);
    

    gettimeofday(&end, NULL);

    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    
    vector<result_t[Range_Size]> values_range(keys_size);
    
    gettimeofday(&start, NULL);
    
    p_size = keys_size / SWP_P2;
    
    // for PPItree only for now 
    // and this PPITree must use sequence leaf node.(one opt for our update is don't use sequence leafnode, only internal node. )
    if (tree.isPPItree) {
        //#pragma omp parallel for 
       // for (int k=0; k<p_size;k++){
        
       //     //range traverse
       //     traverse_PPItree( &(keys[k*SWP_P2]),&(end_keys[k*SWP_P2]), &(leafnode_id[k*SWP_P2]),&(relist_idx[k*SWP_P2]), &(values_range[k*SWP_P2]),tree);
       // }
       return;
    }
    else {
        #pragma omp parallel for 
        for (int k=0; k<p_size;k++){
        
            //range traverse
            traverse_HB( &(keys[k*SWP_P2]),&(end_keys[k*SWP_P2]), &(leafnodes[k*SWP_P2]),&(relist_idx[k*SWP_P2]), &(values_range[k*SWP_P2]));
        }  
    }

    gettimeofday(&end, NULL);
    
    double traverse_time = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        
        cout<<"begin: "<<keys[i]<<" ans: "<<endl;
        for (int j=0; j<Range_Size;j++)
            cout<<values_range[i][j].key<<" ";
        cout<<endl;
        //cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: cpu prefetch sort avxNTG  "<<endl;
    else cout<<"HB+ range [no gpu]: cpu prefetch"<<endl;
    cout<<"Search_size: "<<keys_size<<endl;
    //cout<<"Sort time:      "<<sort_time<<endl;
    cout<<"Search time:      "<<tt<<endl;
    cout<<"Traverse time:      "<<traverse_time<<endl;
    cout<<"Total time:      "<<tt+traverse_time<<endl;
 
/*}}}*/


}




#ifdef TREE_32 

void traverse_HB(key_t *start, key_t *end,BLeaf_node** bleaf_nodes,int *relist_idx, result_t(* values_range)[8]){
}
#else 

void traverse_HB(key_t *start, key_t *end,BLeaf_node** bleaf_nodes,int *relist_idx, result_t(* values_range)[8]){
    for(int i=0;i<SWP_P2;i++){
        key_t start1 = start[i];
        key_t end1 = end[i];
        
        int relist_id=relist_idx[i];
        BLeaf_node * node=bleaf_nodes[i];
        if(node==NULL){
            perror("leaf node is NULL...\n");
            return;
        }
        int used = node->used_relist_slot_num;
 
        int idx=0;
        bool start_flag = 0;
        int ans_num = 0;

        result_t (&results)[Range_Size] = values_range[i];

        while(ans_num<Range_Size){
            if(node==NULL) break;
            key_t key = node->relist[relist_id].r[idx].r_key;
            if(key==Max_Key){
                goto next_position2;
            }

            if(start_flag==0){
                if(key< start1) goto next_position;
                start_flag=1;
            }
            if(key> end1) break;
            results[ans_num].key=key;
            results[ans_num].val=node->relist[relist_id].r[idx].val;
            ans_num++;
next_position:
            idx++;
            if(idx==L_Fanout){
next_position2:
                idx=0;
                relist_id++;
                if (relist_id  >= used ) {
                    relist_id = 0;
                  //  bleaf_pos += 1;          // node's next; BLeaf_node is sorted.
                  //  if (bleaf_pos >= bleaf_count) break;
                    node=node->next;
                    if(node==NULL) break;
                    // node = d_bleafnode + bleaf_pos;
                    used = node->used_relist_slot_num;
                }
            }
        }
    }
}
#endif


#ifdef TREE_32


void search_cpu_prefetch_HB_wo_leaf(key_t* keys,BLeaf_node ** bleafs,int* relist_idx, BPlusTree &tree){
}
#else 



void search_cpu_prefetch_HB_wo_leaf(key_t* keys,BLeaf_node ** bleafs,int* relist_idx, BPlusTree &tree){
/*{{{*/ 
    int height = tree.getHeight();
    void * root=tree.getRoot();
    void *nodes[SWP_P];
    for(int i=0; i<SWP_P;i++) {
        nodes[i]=root;
    }
    //int relist_idx[SWP_P];

    //int SWP_P_div_4 = SWP_P / 4;
    for(int step = 1;step<height; step++){
        for(int i=0;i<SWP_P;i++){
            relist_idx[i]=((Inner_node *)nodes[i])->getChildIdx(NULL,keys[i]);
            nodes[i] = ((Inner_node *)nodes[i])->child[relist_idx[i]];
                __builtin_prefetch(nodes[i],0,3);
            }
        }
    
    for(int i=0;i<SWP_P;i++) bleafs[i]=(BLeaf_node*)nodes[i];
/*}}}*/
    
}

#endif






#endif
