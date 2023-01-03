#include "bpt.h"
#include "conf.h"
#include <iostream>
#include <cassert>
using namespace std;
//-----------------------------------------------------------split aux function
key_t BPlusTree::bleaf_inner_split(BLeaf_node *blfnode,key_t key,value_t val,int relist_idx){
    //move leaf which behind relist_idx backward to get the space for split node 
    if(relist_idx < blfnode->used_relist_slot_num - 1){/*{{{*/
        int end_idx = relist_idx + 1;
        int idx = blfnode->used_relist_slot_num-1;
        for(;idx>=end_idx;idx--){
           blfnode->relist[idx+1] = blfnode->relist[idx];
           blfnode->size[idx+1] = blfnode->size[idx];
        }    
    }
    blfnode->used_relist_slot_num++;
    //
    int point = L_Fanout /2-1;//1
    bool place_right = blfnode->relist[relist_idx].getReordAtIndex(point).r_key < key;
    if(place_right) point++;

    blfnode->relist[relist_idx+1] = Record_list();// clear the list for debug
    for(int i=point;i<L_Fanout;i++){
        blfnode->relist[relist_idx+1].putReordAtIndex(i-point, blfnode->relist[relist_idx].getReordAtIndex(i));
    }
    for(int i=point;i<L_Fanout;i++)blfnode->relist[relist_idx].putReordAtIndex(i,make_record(Max_Key,0));//clear the record for debug

    blfnode->size[relist_idx] = point;
    blfnode->size[relist_idx+1] = L_Fanout-point;
   if(place_right)
       insert_leaf_no_split(blfnode,key,val,relist_idx+1);
   else
       insert_leaf_no_split(blfnode,key,val,relist_idx);
   
   return blfnode->relist[relist_idx+1].getReordAtIndex(0).r_key;/*}}}*/

}
key_t BPlusTree::bleaf_split_core(BLeaf_node *blfnode,BLeaf_node *& newblfnode,key_t key,value_t val,int  relist_idx){
    /*{{{*/ 
    //copy half node to newNode
    int point = BL_leafnum/2;//32
    bool place_right = relist_idx>=point;
    
    std::copy(blfnode->relist+point,blfnode->relist+BL_leafnum,newblfnode->relist);

    for(int i=point;i<BL_leafnum;i++)  blfnode->relist[i]= Record_list();//clear for debug

    std::copy(blfnode->size+point,blfnode->size+BL_leafnum,newblfnode->size);
    for(int i=point;i<BL_leafnum;i++) blfnode->size[i]=0;//clear for debug
    
    blfnode->used_relist_slot_num = point;
    newblfnode->used_relist_slot_num = point;

    newblfnode->next = blfnode->next;
    blfnode->next = newblfnode;
    if(!place_right)
        return bleaf_inner_split(blfnode,key,val,relist_idx);
    else
        return bleaf_inner_split(newblfnode,key,val,relist_idx-(point));/*}}}*/

}
key_t BPlusTree::bleaf_split(BLeaf_node *blfnode,BLeaf_node *& newblfnode,key_t key,value_t val,int  relist_idx){
   /*{{{*/ 
    if(isPPItree)
        newblfnode = bleafNodeCreate_PPItree(blfnode);
    else 
        newblfnode = bleafNodeCreate();     
    return bleaf_split_core(blfnode,newblfnode,key,val,relist_idx);/*}}}*/

}

key_t BPlusTree::inner_node_split_core(Inner_node *inode,Inner_node *& newInode,key_t key,void *oldNextLevelnode,void *nextLevelnode,bool isNormalInnerNode){
    /*{{{*/
    Inner_meta *newImeta = metaOfInnerNodeWithAddress(newInode);

    Inner_meta *oldImeta = metaOfInnerNodeWithAddress(inode);
    //copy half node to newNode
    int point = BPT_ORDER/2;//32
    int idx_point = point/_inner_indexTokey_fanout;//4

    std::copy(inode->inner_key+point,inode->inner_key+BPT_ORDER,newInode->inner_key);
    for(int i=point;i<BPT_ORDER;i++) inode->inner_key[i] = Max_Key;//clear for debug
    key_t ret = inode->inner_key[point-1] ;
    inode->inner_key[point-1] = Max_Key;
   
    
    for(int i=point;i<BPT_ORDER;i++){
        newInode->child[i-point] = inode->child[i];
        inode->child[i] = NULL;
    }
    std::copy(inode->inner_index+idx_point,inode->inner_index+ _inner_indexTokey_fanout,newInode->inner_index);
    for(int i=idx_point;i<_inner_indexTokey_fanout;i++) inode->inner_index[i] = Max_Key;//clear for debug
    inode->inner_index[idx_point-1]  =Max_Key;

    newImeta->index_size = oldImeta->index_size-idx_point;
    oldImeta->index_size = idx_point-1;

    newImeta->parent = oldImeta->parent;

    newImeta->size = oldImeta->size - point;
    oldImeta->size = point-1;
    
    void *child = NULL;
    bool place_right=false; 
    //update child info
    if(leaf_pool.inPool(nextLevelnode)){
        for(int i=0;i<= newImeta->size;i++) newInode->setChild(nextLevelnode,i);    
        BLeaf_node *blnode_nextLevelnode = (BLeaf_node *)nextLevelnode;

        place_right = key >= blnode_nextLevelnode->relist[0].keyAtIndex(0);
        
        if(!place_right){
            child = oldNextLevelnode;
        }else{
            child = nextLevelnode;
        }
    }else{
        if(isNormalInnerNode){
            //PPItree cannot use child pointer to get child
            for(int i=0;i<= newImeta->size;i++) {
                Inner_node *node = (Inner_node *)newInode->child[i];
                metaOfInnerNodeWithAddress(node)->parent = newInode;
            }
        }
        //place_right = key >= ((Inner_node*)nextLevelnode)->inner_key[0];
        place_right = key>=ret;
        if(place_right)
            metaOfInnerNodeWithAddress((Inner_node*)nextLevelnode)->parent = newInode;//parent 问题
        else
            metaOfInnerNodeWithAddress((Inner_node*)nextLevelnode)->parent = inode;//parent 问题
             
        child = nextLevelnode;
    }
    if(place_right){
        insert_inner_no_split(newInode,key,child);//key 问题
    }else{
        insert_inner_no_split(inode,key,child);
    }
    //for ppi tree update 
    if(!isNormalInnerNode&& !leaf_pool.inPool(nextLevelnode)){
        updateChildernInfo(inode,newInode);//update the childern's parent pointer and prefixArray 
    }
    return ret;
/*}}}*/


}
key_t BPlusTree::inner_node_split(Inner_node *inode,Inner_node *& newInode,key_t key,void *oldNextLevelnode,void *nextLevelnode){
    /*{{{*/
    if(isPPItree){
        newInode = INodeCreate_PPItree(inode);
        return inner_node_split_core(inode,newInode,key,oldNextLevelnode,nextLevelnode,false);
    }else{ 
        newInode = INodeCreate();     
        return inner_node_split_core(inode,newInode,key,oldNextLevelnode,nextLevelnode,true);
    }
    /*}}}*/
}


//------------------------------------------------------------ Create aux function 
Inner_node *BPlusTree::INodeCreate(){
   Inner_node *node = inner_node_pool.getElement();

   assert(node !=NULL);

   //Change 0 TO Max_key
   {
        for(int i=0;i<Idxnum_inner;i++) node->inner_index[i] = Max_Key; 
        for(int i=0;i<BPT_ORDER;i++) node->inner_key[i] = Max_Key;
   }
   IMetaCreate();
   return node;
}
Inner_meta *BPlusTree::IMetaCreate(){
    Inner_meta *meta = inner_meta_pool.getElement();
    assert(meta !=NULL);
    meta->index_size = 0;
    meta->size = 0;
    meta->parent = 0;
    return meta;
}
BLeaf_node *BPlusTree::bleafNodeCreate(){
    BLeaf_node *blfnode = NULL;
    
    blfnode = leaf_pool.getElement();

    assert(blfnode != NULL);

    //Change 0 TO Max_key
    for(int i=0;i<BL_leafnum;i++){
        for(int j=0;j<L_Fanout;j++){
            blfnode->relist[i].r[j].r_key = Max_Key;
        }
    }

    blfnode->next = NULL;
    blfnode->parent = NULL;
    blfnode->used_relist_slot_num = 0;

    //for update
    pthread_mutex_init(&(blfnode->lock),NULL);

    return blfnode;
}

BLeaf_node *BPlusTree::bleafNodeCreate(key_t key,value_t val){
    
    BLeaf_node *blfnode = isPPItree ? bleafNodeCreate_PPItree(NULL): bleafNodeCreate();

    blfnode->relist[0].putReordAtIndex(0, make_record(key,val));
    blfnode->next = NULL;
    blfnode->parent = NULL;
    blfnode->size[0] = 1;
    blfnode->used_relist_slot_num = 1;
    return blfnode;
}

//------------------------------------------------------------ Search aux function 

Inner_node *BPlusTree::search_last_level_inner_node(key_t key){
    if(height == 1) return NULL;
    Inner_node *node = (Inner_node *)root;
    int tmp_height = height;
    while(tmp_height > 2){
        Inner_meta *meta = NULL;
        if (key==Max_Key) 
            meta = metaOfInnerNodeWithAddress(node);
        //search index;
        node =(Inner_node *) node->getChild(meta,key);
        tmp_height--;  
    }
    return node;//last inner node 
}
value_t BPlusTree::search_leaf(Inner_node* last_inner_node,key_t key,int &relist_idx,BLeaf_node *&blfnode){
    blfnode =NULL;
    if(last_inner_node==NULL){ 
        blfnode = (BLeaf_node *)root;
        relist_idx = 0;
    }else {
        if (key==Max_Key) 
            relist_idx = last_inner_node->getChildIdx( metaOfInnerNodeWithAddress(last_inner_node),key);
        else
            relist_idx = last_inner_node->getChildIdx( NULL,key);
        blfnode = (BLeaf_node*)last_inner_node->child[0];
    }
    return blfnode->findKey(relist_idx,key);
}

value_t BPlusTree::searchAndUpdate_leaf(Inner_node* last_inner_node,key_t key,int &relist_idx,BLeaf_node *&blfnode, value_t v){
    blfnode =NULL;
    if(last_inner_node==NULL){ 
        blfnode = (BLeaf_node *)root;
        relist_idx = 0;
    }else {
        if (key==Max_Key) 
            relist_idx = last_inner_node->getChildIdx( metaOfInnerNodeWithAddress(last_inner_node),key);
        else
            relist_idx = last_inner_node->getChildIdx( NULL,key);
        blfnode = (BLeaf_node*)last_inner_node->child[0];
    }
    return blfnode->findKeyAndUpdate(relist_idx,key,v);
}



int BPlusTree::search_leaf_fuzzy(Inner_node* last_inner_node,key_t key,int &relist_idx,BLeaf_node *&blfnode, int boundUpOrLow /*low:0 up:1*/){
    blfnode =NULL;
    if(last_inner_node==NULL){ 
        blfnode = (BLeaf_node *)root;
        relist_idx = 0;
    }
    else {
        if (key==Max_Key)
            relist_idx = last_inner_node->getChildIdx( metaOfInnerNodeWithAddress(last_inner_node),key);
        else  
            relist_idx = last_inner_node->getChildIdx( NULL,key);
        blfnode = (BLeaf_node*)last_inner_node->child[0];
    }
    if (boundUpOrLow == 1) return blfnode->findKeyUpperBound(relist_idx, key);
    else return blfnode->findKeyLowerBound(relist_idx,key);
}



//----------------------------------------------------------- Insert  aux function
void BPlusTree::insert_leaf_no_split(BLeaf_node *blfnode,key_t key,value_t val,int relist_idx){
    int &size = blfnode->size[relist_idx];//cur record list size; the size must smaller than 3
    
    assert(size<L_Fanout);
    
    Record_list &relist = blfnode->relist[relist_idx];//cur record list
    int i = size-1;

    for(;i>=0 && relist.keyAtIndex(i)>=key ;i--){//move data
            relist.putReordAtIndex(i+1 , relist.getReordAtIndex(i));
    }
    relist.putReordAtIndex(i+1 , make_record(key,val));//put data
    size++;
    
    assert(size<=L_Fanout);
};

void BPlusTree::insert_inner_no_split(Inner_node *inode,key_t newkey,void *nextLevelnode){
    assert(inode !=NULL);
    Inner_meta *imeta = metaOfInnerNodeWithAddress(inode);
    
    int i=imeta->size;
    for(i=imeta->size-1;i>=0;i--){
        if(newkey < inode->inner_key[i]){
           inode->moveKeyBackward(i,imeta);            
        }else{
            break;
        }
    }
    assert(i>=-1);
    inode->putKeyAndChild(newkey,nextLevelnode,i+1,imeta);
    imeta->size++;
}
void BPlusTree::insert_parent(Inner_node *parent,key_t new_key,void *oldNextLevelnode,void *newNextLevelnode){
/*{{{*/
    if(parent){
        
        if(check_inner_full(parent)){//parent is full ? last_leaf_node is full or inner node is full
            Inner_node *newParent = NULL;
            key_t  key = inner_node_split(parent,newParent,new_key,oldNextLevelnode,newNextLevelnode); 
            
            insert_parent(metaOfInnerNodeWithAddress(parent)->parent,key,parent,newParent);

        }else{
            insert_inner_no_split(parent,new_key,newNextLevelnode);//nextLevelnode = bLeafnode;
        }

    }else{
        if(isPPItree) {
            cout<<"New Level"<<endl;
            exit(0);
        }
        Inner_node *inode = INodeCreate();
        inode->setChild(oldNextLevelnode,0);
        insert_inner_no_split(inode,new_key,newNextLevelnode);
        if(height > 1)//must inner_node 
        {
            metaOfInnerNodeWithAddress((Inner_node *)oldNextLevelnode)->parent = inode;
            metaOfInnerNodeWithAddress((Inner_node *)newNextLevelnode)->parent = inode;
        }
        root = inode;
        height++;
        return;
    }
/*}}}*/
}


//------------------------------------------------------------ Public interface
bool BPlusTree::insert(key_t key, value_t val){
/*{{{*/
    if(root == NULL){ //new Leaf_node,height ==0
        root = (void *)bleafNodeCreate(key,val); 
        
        assert(root!=NULL);
        
        size = height = 1;
        return true;
    }
  
    int relist_idx; 
    Inner_node *last_inner_node = isPPItree ? search_last_level_inner_node_PPItree(key):search_last_level_inner_node(key);
    
    BLeaf_node *blfnode=NULL; 
    value_t idx = isPPItree? search_leaf_PPItree(last_inner_node,key,relist_idx,blfnode) : search_leaf(last_inner_node,key,relist_idx,blfnode); 
    
    if(idx>=0){
        //std::cout<<key<<" exist!"<<std::endl;
        return false;
    }
    assert(blfnode!=NULL);
    
    if(!check_leaf_full(blfnode,relist_idx)){
        insert_leaf_no_split(blfnode,key,val,relist_idx);
        return true;
    }
    
    //do split
    int next_slot = -1;
    key_t new_key = -1;
    if((next_slot = blfnode->bleaf_has_slot())!=-1){//this big node is not full
      new_key =  bleaf_inner_split(blfnode,key,val,relist_idx);//relist_idx is change to new slot 
      // insert into parent 
      insert_parent(last_inner_node,new_key,blfnode,blfnode); 
      return true;
    }

    // new a big node 
    BLeaf_node *newblfnode = NULL;
    new_key = bleaf_split(blfnode,newblfnode,key,val,relist_idx);

    //insert to parent
    insert_parent(last_inner_node,new_key,blfnode,newblfnode);

    //if(isPPItree) {
    //    cout<<key<<endl;
    //    check();//check parent and child info 
    //}

    return true;
/*}}}*/
}
//============================================================================================================= Parallel Update
static void* thread_update_parallel(void *args){
    Parallel_update_data *data = (Parallel_update_data *)args;/*{{{*/
    int my_idx = data->thread_idx;;
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
        key_t key = keys[my_idx];
        value_t val = vals[my_idx];
        Inner_node *last_inner_node = tree->isPPItree? tree->search_last_level_inner_node_PPItree(key): tree->search_last_level_inner_node(key);
    
        BLeaf_node *blfnode=NULL; 
        
        if(tree->isPPItree){
            int blfnode_idx= tree->getInnerNodeIdx(last_inner_node)- tree->getInnerSize_wo_last_inner_node();
            blfnode =  tree->getBLeafNodeByIdx(blfnode_idx);
        }else{
            blfnode = (BLeaf_node *) last_inner_node->child[0];
        }

        pthread_mutex_lock(&blfnode->lock);
        value_t idx = tree->isPPItree? tree->searchAndUpdate_leaf_PPItree(last_inner_node,key,relist_idx,blfnode,key-1): tree->searchAndUpdate_leaf(last_inner_node,key,relist_idx,blfnode, key-1); //search and if exists, update value =  (key-1) 
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
        idx = tree->isPPItree? tree->search_leaf_PPItree(last_inner_node,key,relist_idx,blfnode): tree->search_leaf(last_inner_node,key,relist_idx,blfnode); 
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
/*    
        if(!tree->check_leaf_full(blfnode,relist_idx)){
            pthread_mutex_lock(&blfnode->lock);
            idx = tree->isPPItree? tree->search_leaf_PPItree(last_inner_node,key,relist_idx,blfnode): tree->search_leaf(last_inner_node,key,relist_idx,blfnode); 
            if(!tree->check_leaf_full(blfnode,relist_idx)){
                tree->insert_leaf_no_split(blfnode,key,val,relist_idx);
                pthread_mutex_unlock(&blfnode->lock);
                isInsertArr[my_idx] = 1;
                my_idx+=thread_num;
                continue;
            }else{
                pthread_mutex_unlock(&blfnode->lock);

            }
        }
    
         //do inner leaf_node split
        int next_slot = -1;
        key_t new_key = -1;

        if((next_slot = blfnode->bleaf_has_slot())!=-1){//this big node is not full
            pthread_mutex_lock(&blfnode->lock);
            if((next_slot = blfnode->bleaf_has_slot())!=-1){//this big node is not full

                value_t idx = tree->isPPItree? tree->search_leaf_PPItree(last_inner_node,key,relist_idx,blfnode):tree->search_leaf(last_inner_node,key,relist_idx,blfnode); 
                if(!tree->check_leaf_full(blfnode,relist_idx)){
                    tree->insert_leaf_no_split(blfnode,key,val,relist_idx);
                    isInsertArr[my_idx] = 1;
                    pthread_mutex_unlock(&blfnode->lock);
                    my_idx+=thread_num;
                    continue;
                }
                new_key =  tree->bleaf_inner_split(blfnode,key,val,relist_idx);//relist_idx is change to new slot 
                // insert into parent 
                tree->insert_parent(last_inner_node,new_key,blfnode,blfnode); 
                isInsertArr[my_idx] = 1;
            }
            pthread_mutex_unlock(&blfnode->lock);
            my_idx+=thread_num;
            continue;
        }
*/
        my_idx += thread_num;
    }/*}}}*/
}
bool BPlusTree::insert_parallel(const vector<key_t> &keys,const vector<value_t> &vals){
    int batch_size  = keys.size();/*{{{*/
    int thread_num = CPU_LOGICAL_CORE;
    vector<int> isInsertArr(batch_size,0);
    Parallel_update_data parallel_update_data;
    parallel_update_data.keys = &keys;
    parallel_update_data.vals = &vals;
    parallel_update_data.isInsertArr = &isInsertArr;
    parallel_update_data.batch_size = batch_size;
    parallel_update_data.thread_num = thread_num;
    parallel_update_data.bpt = this;

    //parallel update
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
    //update left keys 
    int count = 0;
    int repeat  = 0;
    for(int i=0;i<batch_size;i++){
       
        if(isInsertArr[i]==1) continue;
        if(isInsertArr[i]==-1){
            //cout<<"repeat:"<<keys[i]<<endl;
            repeat++;
            continue;
        }
        count++;
       insert(keys[i],vals[i]);
    }
    cout<<"repeat num: "<<repeat<<endl;
    cout<<"serial update num: "<<count<<endl;
#ifdef ENABLE_TEST
    cout<<"Start search"<<endl;
    for(int i=0;i<keys.size();i++){
        value_t v = search(keys[i]);
        //if(v==-1)
            cout<<i<<" "<<keys[i]<<  " " <<v<<endl;
    }
    cout<<"End search"<<endl;
#endif
    return true;/*}}}*/
}

bool BPlusTree::insert_parallel_Newupdate(const vector<key_t> &keys,const vector<value_t> &vals){
    //same as insert parallel ,becauce no split/*{{{*/
    int batch_size  = keys.size();
    int thread_num = CPU_LOGICAL_CORE;
    vector<int> isInsertArr(batch_size,0);
    Parallel_update_data parallel_update_data;
    parallel_update_data.keys = &keys;
    parallel_update_data.vals = &vals;
    parallel_update_data.isInsertArr = &isInsertArr;
    parallel_update_data.batch_size = batch_size;
    parallel_update_data.thread_num = thread_num;
    parallel_update_data.bpt = this;

    //parallel update
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

    //cout<<"Start:Parallel_update check"<<endl;
  	//for(int i=0;i<batch_size;i++){
    //    if(isInsertArr[i]==1) {
    //        value_t v = search_PPItree_Newupdate(keys[i]);
    //        if(v==-1)
    //          cout<<i<<" "<<keys[i]<<  " " <<v<<endl;
    //    }
    //    if(isInsertArr[i]==-1){
    //        cout<<"repeat:"<<keys[i]<<endl;
    //        continue;
    //    }
    //}
    //cout<<"End:Parallel_update check"<<endl;

    //different with insert_parallel
    //update left keys 
    int count = 0;
    int repeat  = 0;
    for(int i=0;i<batch_size;i++){
       
        if(isInsertArr[i]==1) continue;
        if(isInsertArr[i]==-1){
            repeat++;
            //cout<<"repeat:"<<keys[i]<<endl;
            continue;
        }
        count++;
        insert_Newupdate(keys[i],vals[i]);

	   // cout<<"serial update num: "<<count<<endl;
	   //    cout<<"Start search"<<endl;
	   //    for(int j=0;j<=i;j++){
	   //        //if(j==51825) 
	   //        //    cout<<"eha"<<endl;
	   //         value_t v = search_PPItree_Newupdate(keys[j]);
	   //        if(v==-1){
	   //            cout<<j<<" "<<keys[j]<<  " " <<v<<endl;
	   //            exit(0);
	   //        }
	   //    }

	   //    cout<<"End search"<<endl;

    }


    cout<<"repeat num: "<<repeat<<endl;
    cout<<"serial update num: "<<count<<endl;
    if (count!=0) {
        cout<<"rebuild start "<<endl;
        //rebuild(); 
        rebuild_without_leaf(); 
    }
  
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

//============================================================================================================ 
value_t BPlusTree::search(key_t key){

    Inner_node *last_inner_node = isPPItree?search_last_level_inner_node_PPItree(key): search_last_level_inner_node(key);
    
    int relist_idx;
    BLeaf_node *blfnode = NULL;
    return  isPPItree? search_leaf_PPItree(last_inner_node,key,relist_idx,blfnode): search_leaf(last_inner_node, key, relist_idx, blfnode);
}
void BPlusTree::search_swp(const vector<key_t> &keys, vector<value_t> &vals){
    void * nodes[P];//P=16
    for(int i=0;i<P;i++) nodes[i] = root;

    int relist_idx[P];
    for(int step = 1;step<height;step++){
        for(int i = 0;i<P;i++){
             relist_idx[i]=((Inner_node *)nodes[i])->getChildIdx_avx2(NULL,keys[i]);//avx or not?
             nodes[i] = ((Inner_node *)nodes[i])->child[relist_idx[i]];
             __builtin_prefetch(nodes[i],0,3);
        }
    }
    for(int i=0;i<P;i++) vals[i] = ((BLeaf_node*)nodes[i])->findKey(relist_idx[i],keys[i]);
}
void BPlusTree::range_search(key_t start_key,key_t end_key, vector<Record> *v){
    if (start_key>end_key) return;
    Inner_node *last_inner_node_start = search_last_level_inner_node(start_key);
    Inner_node *last_inner_node_end = search_last_level_inner_node(end_key);
    //cout<<"start_inner_node: "<<last_inner_node_start<<" end_inner_node: "<<last_inner_node_end<<endl;

    BLeaf_node *blfnode_start = NULL;
    BLeaf_node *blfnode_end = NULL;
    int relist_idx_start;
    int relist_idx_end;
    
    int idx_start = search_leaf_fuzzy(last_inner_node_start, start_key, relist_idx_start, blfnode_start, 0);
    int idx_end = search_leaf_fuzzy(last_inner_node_end, end_key, relist_idx_end, blfnode_end, 1);
    
   // int idx_start = search_leaf(last_inner_node_start, start_key, relist_idx_start, blfnode_start);
   // int idx_end = search_leaf(last_inner_node_end, end_key, relist_idx_end, blfnode_end);
    
   /* 
    cout<<"idx_start: "<<idx_start<<" idx_end: "<<idx_end<<endl;
    cout<<"relist_idx_start: "<<relist_idx_start<<" relist_idx_end: "<<relist_idx_end<<endl;
    cout<<"blfnode_start: "<<blfnode_start<<" blfnode_end: "<<blfnode_end<<endl;
    */
     
    if (idx_start==-1) {
        idx_start = 0;
        relist_idx_start++;

        if (relist_idx_start>= blfnode_start->used_relist_slot_num){
            if (blfnode_start==blfnode_end) return;
            
            blfnode_start = blfnode_start->next;
            relist_idx_start = 0;
        }
    }
   

    if (idx_end==-1) {
        relist_idx_end--;
        if (relist_idx_end<0){

            if (blfnode_start==blfnode_end) return;
           
            BLeaf_node *temp = blfnode_start;
            BLeaf_node *temp1 = blfnode_start;

            while (temp!=blfnode_end) {
                temp1 = temp;
                temp = temp->next;
            }
            blfnode_end = temp1;    //get front
            relist_idx_end = -1;
        }

    }
   /* 
    cout<<"idx_start: "<<idx_start<<" idx_end: "<<idx_end<<endl;
    cout<<"relist_idx_start: "<<relist_idx_start<<" relist_idx_end: "<<relist_idx_end<<endl;
    cout<<"blfnode_start: "<<blfnode_start<<" blfnode_end: "<<blfnode_end<<endl;
    */
    
    if (blfnode_start == blfnode_end) {

        blfnode_start->getKeyValuesRange(relist_idx_start, relist_idx_end, idx_start, idx_end, v);
        return;
    }
    
    blfnode_start->getKeyValuesRange(relist_idx_start, -1, idx_start, -1,v);
    blfnode_start = blfnode_start->next;
    for (; blfnode_start != blfnode_end; blfnode_start = blfnode_start->next) {
        if (blfnode_start == NULL) return;
        blfnode_start->getKeyValuesRange(0, -1, 0, -1, v); 
    }
    blfnode_end->getKeyValuesRange(0,relist_idx_end, 0, idx_end, v);
    return;
}

bool BPlusTree::remove(key_t key){

}
