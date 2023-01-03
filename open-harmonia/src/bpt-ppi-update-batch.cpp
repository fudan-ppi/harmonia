#include "bpt.h"
#include "conf.h"
#include <iostream>
#include <cassert>
using namespace std;




static void* thread_update_parallel(void *args){
    Parallel_update_data *data = (Parallel_update_data *)args;/*{{{*/
    int my_idx = data->thread_idx;
    int offset = data->offset;
    int thread_num = data->thread_num;
    BPlusTree *tree = data->bpt;
    vector<int> &isInsertArr = *(data->isInsertArr) ;
    const vector<key_t> &keys = *(data->keys) ;
    const vector<value_t> &vals = *(data->vals);

    int relist_idx; 
    while(my_idx < data->batch_size ){
        // same as insert funtion front part
        if(tree->getRoot() == NULL){ //new Leaf_node,height ==0
            //isInsertArr[my_idx] = false;
            my_idx+=thread_num;
            continue;
        }
        key_t key = keys[my_idx+offset];
        value_t val = vals[my_idx+offset];
        Inner_node *last_inner_node = tree->search_last_level_inner_node_PPItree_Newupdate(key);
    
        BLeaf_node *blfnode=NULL; 
        
        int last_inner_node_idx = tree->getInnerNodeIdx(last_inner_node);

        if (tree->isNormalInnerNode(last_inner_node_idx)) {
            blfnode = (BLeaf_node *) last_inner_node->child[0];
        }
        else{
            int blfnode_idx= tree->getInnerNodeIdx(last_inner_node)- tree->getInnerSize_wo_last_inner_node();
            blfnode =  tree->getBLeafNodeByIdx(blfnode_idx);
        }
// TODO: notice: here we already get blfnode
// but functions like searchAndUpdate_leaf will also use last inner node to get blfnode one more time. which is a waste in this situation.
// we can just use a new function which is uncoupled to fix this.
        pthread_mutex_lock(&blfnode->lock);
        value_t idx = tree->searchAndUpdate_leaf_PPItree_Newupdate(last_inner_node,key,relist_idx,blfnode,key-1); //search and if exists, update value =  (key-1) 
        pthread_mutex_unlock(&blfnode->lock);

        if(idx>=0){
            //std::cout<<key<<" exist!"<<std::endl;
            //isInsertArr[my_idx] = false;
            isInsertArr[my_idx] = -1;
            my_idx+=thread_num;
            continue;
        }
        
        assert(blfnode!=NULL);
        pthread_mutex_lock(&blfnode->lock);
        idx = tree->search_leaf_PPItree_Newupdate(last_inner_node, key, relist_idx, blfnode);
        if(!tree->check_leaf_full(blfnode,relist_idx)){
            tree->insert_leaf_no_split(blfnode,key,val,relist_idx);
            pthread_mutex_unlock(&blfnode->lock);
            isInsertArr[my_idx] = 1;
            my_idx+=thread_num;
            continue;
        }
        int next_slot = -1;
        key_t new_key = -1;
        
        if((next_slot = blfnode->bleaf_has_slot())!=-1){//this big node is not full
            new_key =  tree->bleaf_inner_split(blfnode,key,val,relist_idx);//relist_idx is change to new slot 
                // insert into parent 
            tree->insert_parent(last_inner_node,new_key,blfnode,blfnode); 
            isInsertArr[my_idx] = 1;
        }
        pthread_mutex_unlock(&blfnode->lock);

        my_idx += thread_num;
    }/*}}}*/
}



//two batch and rebuild without leaf
bool BPlusTree::insert_parallel_Newupdate_batch(const vector<key_t> &keys,const vector<value_t> &vals, int syn_batch_size, int rebuild_batch_size){
    //same as insert parallel ,becauce no split/*{{{*/

    int queries_num = keys.size(); 
    int dealed_queries_num = 0;

    int thread_num = CPU_LOGICAL_CORE;

    Parallel_update_data parallel_update_data;
    parallel_update_data.keys = &keys;
    parallel_update_data.vals = &vals;
    parallel_update_data.batch_size = syn_batch_size;
    parallel_update_data.thread_num = thread_num;
    parallel_update_data.bpt = this;

    int serial_count = 0;
    int tmp_serial_count = 0;
    int repeat = 0;

    while (dealed_queries_num < queries_num) {
       
        
        vector<int> isInsertArr(syn_batch_size,0);
        parallel_update_data.isInsertArr = &isInsertArr;
        parallel_update_data.offset = dealed_queries_num;

        //----------------parallel update---------------------------------------
        vector<pthread_t> tid_arr;
        vector<Parallel_update_data> parallel_update_data_arr(thread_num,parallel_update_data);
        pthread_t tid;

        for(int i=0; i<thread_num;i++){
            parallel_update_data_arr[i].thread_idx = i;
            if(pthread_create(&tid,NULL,thread_update_parallel,(void *)&parallel_update_data_arr[i])!=0){
                cout<<"Cant create thread"<<endl;
            }else{
               tid_arr.push_back(tid); 
            }
        }

        
        //wait for parallel update
        for(int i=0;i<thread_num;i++){
            pthread_join(tid_arr[i],NULL);
        }
  
#ifdef ENABLE_TEST
        cout<<"Start:Parallel_update check"<<endl;
        for(int i=0;i<syn_batch_size;i++){
            int ii = i + dealed_queries_num;
            if(isInsertArr[i]==1) {
                value_t v = search_PPItree_Newupdate(keys[ii]);
                if(v==-1)
                  cout<<i<<" "<<keys[ii]<<  " " <<v<<endl;
            }
            if(isInsertArr[i]==-1){
                cout<<"repeat:"<<keys[ii]<<endl;
                continue;
            }
        }
        cout<<"End:Parallel_update check"<<endl;
#endif        
        //----------------serial------------------------------
        
//        cout<<"serial"<<endl; 
        for(int i=0;i<syn_batch_size;i++){
            int ii = i + dealed_queries_num; 
            if(isInsertArr[i]==1) continue;
            if(isInsertArr[i]==-1){
                repeat++;
                //cout<<"repeat:"<<keys[i]<<endl;
                continue;
            }
            serial_count++;
            insert_Newupdate(keys[ii],vals[ii]);
 
#ifdef ENABLE_TEST
            //serial test
            value_t v = search_PPItree_Newupdate(keys[ii]);
            
            if(v==-1){
                cout<<ii<<" "<<keys[ii]<<  " " <<v<<endl;
            }
#endif
        }

        dealed_queries_num += syn_batch_size;
//        cout<<"serial done"<<endl;   
        //----------------rebuild----------------------------------------------
        if ((dealed_queries_num % rebuild_batch_size == 0)&&(serial_count!=tmp_serial_count)) {
            //cout<<"rebuild start "<<endl;
            //rebuild();
            rebuild_without_leaf();
            tmp_serial_count = serial_count;
        }
        
    }


    cout<<"repeat num: "<<repeat<<endl;
    cout<<"serial update num: "<<serial_count<<endl;
  
#ifdef ENABLE_TEST 
    //not right for rebuild without leaf
    cout<<"Start search"<<endl;
    for(int i=0;i<keys.size();i++){
        value_t v = search_PPItree_Newupdate(keys[i]);
        if(v==-1)
            cout<<i<<" "<<keys[i]<<  " " <<v<<endl;
    }
    cout<<"End search"<<endl;
#endif

    return true;/*}}}*/
}



static void* thread_update_parallel_hb(void *args){
    Parallel_update_data *data = (Parallel_update_data *)args;/*{{{*/
    int my_idx = data->thread_idx;
    int offset = data->offset;
    int thread_num = data->thread_num;
    BPlusTree *tree = data->bpt;
    vector<int> &isInsertArr = *(data->isInsertArr) ;
    const vector<key_t> &keys = *(data->keys) ;
    const vector<value_t> &vals = *(data->vals);

    int relist_idx; 
    while(my_idx < data->batch_size ){
        // same as insert funtion front part
        if(tree->getRoot() == NULL){ //new Leaf_node,height ==0
            //isInsertArr[my_idx] = false;
            my_idx+=thread_num;
            continue;
        }
        key_t key = keys[my_idx+offset];
        value_t val = vals[my_idx+offset];
        Inner_node *last_inner_node = tree->search_last_level_inner_node(key);
    
        BLeaf_node *blfnode=NULL; 
        
        blfnode = (BLeaf_node *) last_inner_node->child[0];
// TODO: notice: here we already get blfnode
// but functions like searchAndUpdate_leaf will also use last inner node to get blfnode one more time. which is a waste in this situation.
// we can just use a new function which is uncoupled to fix this.
        pthread_mutex_lock(&blfnode->lock);
        value_t idx = tree->searchAndUpdate_leaf(last_inner_node,key,relist_idx,blfnode,key-1); //search and if exists, update value =  (key-1) 
        pthread_mutex_unlock(&blfnode->lock);

        if(idx>=0){
            //std::cout<<key<<" exist!"<<std::endl;
            //isInsertArr[my_idx] = false;
            isInsertArr[my_idx] = -1;
            my_idx+=thread_num;
            continue;
        }
        
        assert(blfnode!=NULL);
        pthread_mutex_lock(&blfnode->lock);
        idx = tree->search_leaf(last_inner_node, key, relist_idx, blfnode);
        if(!tree->check_leaf_full(blfnode,relist_idx)){
            tree->insert_leaf_no_split(blfnode,key,val,relist_idx);
            pthread_mutex_unlock(&blfnode->lock);
            isInsertArr[my_idx] = 1;
            my_idx+=thread_num;
            continue;
        }
        int next_slot = -1;
        key_t new_key = -1;
        
        if((next_slot = blfnode->bleaf_has_slot())!=-1){//this big node is not full
            new_key =  tree->bleaf_inner_split(blfnode,key,val,relist_idx);//relist_idx is change to new slot 
                // insert into parent 
            tree->insert_parent(last_inner_node,new_key,blfnode,blfnode); 
            isInsertArr[my_idx] = 1;
        }
        pthread_mutex_unlock(&blfnode->lock);

        
        my_idx += thread_num;
    }/*}}}*/
}




//two batch hb
bool BPlusTree::insert_parallel_batch(const vector<key_t> &keys,const vector<value_t> &vals, int batch_size){


    //same as insert parallel ,becauce no split/*{{{*/

    int queries_num = keys.size(); 
    int dealed_queries_num = 0;

    int thread_num = CPU_LOGICAL_CORE;

    Parallel_update_data parallel_update_data;
    parallel_update_data.keys = &keys;
    parallel_update_data.vals = &vals;
    parallel_update_data.batch_size = batch_size;
    parallel_update_data.thread_num = thread_num;
    parallel_update_data.bpt = this;

    int serial_count = 0;
    int repeat = 0;

    while (dealed_queries_num < queries_num) {
       
        
        vector<int> isInsertArr(batch_size,0);
        parallel_update_data.isInsertArr = &isInsertArr;
        parallel_update_data.offset = dealed_queries_num;

        //----------------parallel update---------------------------------------
        vector<pthread_t> tid_arr;
        vector<Parallel_update_data> parallel_update_data_arr(thread_num,parallel_update_data);
        pthread_t tid;

        for(int i=0; i<thread_num;i++){
            parallel_update_data_arr[i].thread_idx = i;
            if(pthread_create(&tid,NULL,thread_update_parallel_hb,(void *)&parallel_update_data_arr[i])!=0){
                cout<<"Cant create thread"<<endl;
            }else{
               tid_arr.push_back(tid); 
            }
        }

        
        //wait for parallel update
        for(int i=0;i<thread_num;i++){
            pthread_join(tid_arr[i],NULL);
        }
  
#ifdef ENABLE_TEST
        cout<<"Start:Parallel_update check"<<endl;
        for(int i=0;i<batch_size;i++){
            int ii = i + dealed_queries_num;
            if(isInsertArr[i]==1) {
                value_t v = search(keys[ii]);
                if(v==-1)
                  cout<<i<<" "<<keys[ii]<<  " " <<v<<endl;
            }
            if(isInsertArr[i]==-1){
                cout<<"repeat:"<<keys[ii]<<endl;
                continue;
            }
        }
        cout<<"End:Parallel_update check"<<endl;
#endif        
        //----------------serial------------------------------
        
//        cout<<"serial"<<endl; 
        for(int i=0;i<batch_size;i++){
            int ii = i + dealed_queries_num; 
            if(isInsertArr[i]==1) continue;
            if(isInsertArr[i]==-1){
                repeat++;
                //cout<<"repeat:"<<keys[i]<<endl;
                continue;
            }
            serial_count++;
            insert(keys[ii],vals[ii]);
 
#ifdef ENABLE_TEST
            //serial test
            value_t v = search(keys[ii]);
            
            if(v==-1){
                cout<<ii<<" "<<keys[ii]<<  " " <<v<<endl;
            }
#endif
        }

        dealed_queries_num += batch_size;
//        cout<<"serial done"<<endl;   
    }


    cout<<"repeat num: "<<repeat<<endl;
    cout<<"serial update num: "<<serial_count<<endl;
  
#ifdef ENABLE_TEST 
    //not right for rebuild without leaf
    cout<<"Start search"<<endl;
    for(int i=0;i<keys.size();i++){
        value_t v = search(keys[i]);
        if(v==-1)
            cout<<i<<" "<<keys[i]<<  " " <<v<<endl;
    }
    cout<<"End search"<<endl;
#endif

    return true;/*}}}*/




}
