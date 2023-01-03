#include "bpt.h"
#include <iostream>
#include <cassert>
#include <string.h>
#include <queue>
#include <omp.h>
#include "conf.h"
using namespace std;


//search operation
Inner_node * BPlusTree::search_last_level_inner_node_PPItree_Newupdate(key_t key){
   if(height == 1) return NULL;

   int tmp_height = height;
   int inner_node_idx = 0;// tree root's idx

   Inner_node *node = nullptr;
   
   //
   node = getInnerNodeByIdx(inner_node_idx);

   while(tmp_height>2){
        Inner_meta *meta = NULL;
        if(key == Max_Key)
            meta = metaOfInnerNodeWithIdx(inner_node_idx);
        //
        if(isNormalInnerNode(inner_node_idx)){
            node = (Inner_node *)node->getChild(meta,key); 
            inner_node_idx = inner_node_pool.indexAtAddress(node);
        }else{ 
            int child_idx = node->getChildIdx(meta,key);
            inner_node_idx = prefixArray[inner_node_idx]+child_idx;
            node = getInnerNodeByIdx(inner_node_idx);
        }
        tmp_height--;
   }
   return node;
}
value_t BPlusTree::search_leaf_PPItree_Newupdate(Inner_node *last_inner_node,key_t key,int &relist_idx, BLeaf_node *&blfnode){
    blfnode = nullptr;
    if(last_inner_node == nullptr){
        blfnode = (BLeaf_node*) root;
        relist_idx = 0;
    }else{
        if(key==Max_Key){
            relist_idx = last_inner_node->getChildIdx(metaOfInnerNodeWithAddress(last_inner_node),key);
        }else{
            relist_idx = last_inner_node->getChildIdx(nullptr,key);
        }
        //diff
        if(isNormalInnerNode(getInnerNodeIdx(last_inner_node))){
            blfnode = static_cast<BLeaf_node *>(last_inner_node->child[0]);
        }else{
            int blfnode_idx = getInnerNodeIdx(last_inner_node)-getInnerSize_wo_last_inner_node();
            blfnode = getBLeafNodeByIdx(blfnode_idx);
        }
    }
    return blfnode->findKey(relist_idx,key);
}
value_t BPlusTree::searchAndUpdate_leaf_PPItree_Newupdate(Inner_node *last_inner_node,key_t key,int &relist_idx, BLeaf_node *&blfnode,value_t v){
    blfnode = nullptr;
    if(last_inner_node == nullptr){
        blfnode = (BLeaf_node*) root;
        relist_idx = 0;
    }else{
        if(key==Max_Key){
            relist_idx = last_inner_node->getChildIdx(metaOfInnerNodeWithAddress(last_inner_node),key);
        }else{
            relist_idx = last_inner_node->getChildIdx(nullptr,key);
        }
        //diff
        if(isNormalInnerNode(getInnerNodeIdx(last_inner_node))){
            blfnode = static_cast<BLeaf_node *>(last_inner_node->child[0]);
        }else{
            int blfnode_idx = getInnerNodeIdx(last_inner_node)-getInnerSize_wo_last_inner_node();
            blfnode = getBLeafNodeByIdx(blfnode_idx);
        }
    }
    return blfnode->findKeyAndUpdate(relist_idx,key,v);
}


value_t BPlusTree::search_PPItree_Newupdate(key_t key){
    Inner_node *last_inner_node = search_last_level_inner_node_PPItree_Newupdate(key);
    int relist_idx;
    BLeaf_node *blfnode = NULL;
    return search_leaf_PPItree_Newupdate(last_inner_node,key,relist_idx,blfnode);
}

//split operation
key_t BPlusTree::bleaf_split_Newupdate(BLeaf_node *&blfnode, BLeaf_node *&newblfnode,key_t key, value_t val, int relist_idx){
    newblfnode = bleafNodeCreate();
    key_t new_key =  bleaf_split_core(blfnode,newblfnode,key,val,relist_idx);
    return new_key;
}

// insert operation
bool BPlusTree::insert_Newupdate(key_t key,value_t val){
    if(root==NULL){
        root = (void *)bleafNodeCreate(key,val);
        assert(root!=NULL);
        size = height = 1;
        return true;
    }
    int relist_idx;
    Inner_node *last_inner_node = search_last_level_inner_node_PPItree_Newupdate(key);
    BLeaf_node *blfnode = nullptr;

    value_t idx = search_leaf_PPItree_Newupdate(last_inner_node,key,relist_idx,blfnode);
    if(idx>=0){
        //cout<<"exist"<<endl;
        return false;
    }

    assert(blfnode!=NULL);
    
    if(!check_leaf_full(blfnode,relist_idx)){
        insert_leaf_no_split(blfnode,key,val,relist_idx);
        return true;
    }
    
    int new_slot = -1;
    key_t new_key = -1;

    if((new_slot = blfnode->bleaf_has_slot())!=-1){
        new_key = bleaf_inner_split(blfnode,key,val,relist_idx);
        insert_parent(last_inner_node,new_key,blfnode,blfnode);
        return true;
    }


    // differ with operation insert 
    BLeaf_node *new_blfnode = nullptr;
    new_key = bleaf_split_Newupdate(blfnode,new_blfnode,key,val,relist_idx);    
    insert_parent_Newupdate(last_inner_node,new_key,blfnode,new_blfnode);
    return true;
}

void BPlusTree::insert_parent_Newupdate(Inner_node *parent,key_t newkey, void *oldNextLevelNode, void *newNextLevelnode){
    if(parent){
        if(check_inner_full(parent)){

            Inner_node *newParent = INodeCreate();
            changeToNormalInnerNode(getInnerNodeIdx(parent));
            //changeToNormalInnerNode(getInnerNodeIdx(newParent));     
            markAsNormalInnerNode(getInnerNodeIdx(newParent));
            key_t key = inner_node_split_core(parent,newParent,newkey,oldNextLevelNode,newNextLevelnode,true);
            insert_parent_Newupdate(metaOfInnerNodeWithAddress(parent)->parent,key,parent,newParent); 
        }else{
            changeToNormalInnerNode(getInnerNodeIdx(parent));
            insert_inner_no_split(parent,newkey,newNextLevelnode);
        }
    }else{
        //new level 
        cout<<"New Level Error"<<endl;
    }
}

//this method cannot be used on new node.
void BPlusTree::changeToNormalInnerNode(int idx){
    if(!isNormalInnerNode(idx)){
        int size =  metaOfInnerNodeWithIdx(idx)->size;
        Inner_node *innode = getInnerNodeByIdx(idx);

        if(idx<getInnerSize_wo_last_inner_node()){
            for(int i=0;i<=size;i++){
               innode->child[i] = getInnerNodeByIdx(prefixArray[idx]+i); 
            }
        }else{
            BLeaf_node *blfnode = getBLeafNodeByIdx(idx-getInnerSize_wo_last_inner_node());
            innode->child[0] = blfnode;
        }
        markAsNormalInnerNode(idx);
    }
}

typedef struct {
    int idx;
    int height; //root as 0
}q_ele;
void BPlusTree::rebuild(){

    Mem_Pool<Inner_node> tmp_inner_node_pool;
    Mem_Pool<Inner_meta> tmp_inner_meta_pool;
    Mem_Pool<BLeaf_node> tmp_bleaf_node_pool;
    
    int *new_levelNum = (int*)malloc(sizeof(int) * height);
    int *new_levelIdx = (int*)malloc(sizeof(int) * height);
    
    int node_num_new = inner_node_pool.getSize();
    int leaf_num_new = leaf_pool.getSize();
    
    int *new_prefixArray = (int*)malloc(sizeof(int) * node_num_new*2); // x2 for possible grow of prefix, which is not necessary since we use this update method of prefix.

    int *oldToNew = (int *)malloc(sizeof(int) * node_num_new);
    int *oldToNew_leaf = (int *)malloc(sizeof(int) * leaf_num_new); 
    
    for (int i=0;i<height;i++) {
        new_levelNum[i] = 0;
        new_levelIdx[i] = 0;
    }

    int nodeIdx = 0;    //root;
    int inner_size_wo_last_inner_node = getInnerSize_wo_last_inner_node();  //levelIdx[height-2]
    queue<q_ele> inner_node_q;
    inner_node_q.push({nodeIdx,0});             //root as height 0,    para 'height' start from 1.
    
    queue<int> leaf_q;

    
    int pos = -1;
    new_prefixArray[0] = 1;
    while (!inner_node_q.empty()){
  
        pos++;
        
        int nodeIdx = inner_node_q.front().idx;
        int h = inner_node_q.front().height;  
        new_levelNum[h]++;


        Inner_node* n = getInnerNodeByIdx(nodeIdx);
        //cout<<nodeIdx<<" "<<h<<endl;
        
        inner_node_q.pop();

        //memcpy (tmp_inner_node_pool.getElement(),n, sizeof(key_t)*72);
        //memcpy (tmp_inner_meta_pool.getElement(),metaOfInnerNodeWithIdx(nodeIdx), sizeof(Inner_meta));
        oldToNew[nodeIdx] = pos;

        int size = metaOfInnerNodeWithIdx(nodeIdx)->size;   //num of key, not child
        new_prefixArray[pos+1] = new_prefixArray[pos] +size+1; 

        if (isNormalInnerNode(nodeIdx)) {
            if (h==height-2) {      //last inner node

                leaf_q.push(getLeafIdx((BLeaf_node *)n->child[0])); 
            }
            else{
                for (int i=0;i<=size;i++) {
                    inner_node_q.push({getInnerNodeIdx((Inner_node*)n->child[i]), h+1}); 
                }
            }

        }
        else{
            if (h==height-2) {      //last inner node
                leaf_q.push(nodeIdx - inner_size_wo_last_inner_node);
            }
            else{
                for (int i=0;i<=size;i++) {
                    inner_node_q.push({prefixArray[nodeIdx]+i, h+1});
                }
            }
        }
    }
    pos = -1;
    while (!leaf_q.empty()) {
        pos++;
        int nodeIdx = leaf_q.front();
        //memcpy(tmp_bleaf_node_pool.getElement(), getBLeafNodeByIdx(nodeIdx), sizeof(BLeaf_node));
        oldToNew_leaf[nodeIdx] = pos;
        
        leaf_q.pop();
    }
  
    for (int i=0; i<node_num_new;i++) {
        tmp_inner_node_pool.getElement();
        tmp_inner_meta_pool.getElement();
    }
    for (int i=0; i<leaf_num_new;i++) {
        tmp_bleaf_node_pool.getElement();
    }

    //memcpy

    omp_set_num_threads(OMP_Thread_num);
#pragma omp parallel for
    for (int i=0; i<node_num_new;i++) {
        //memcpy (tmp_inner_node_pool.getElement(),n, sizeof(key_t)*72);
        //memcpy (tmp_inner_meta_pool.getElement(),metaOfInnerNodeWithIdx(nodeIdx), sizeof(Inner_meta));
        memcpy(tmp_inner_node_pool.elementAtIdx(oldToNew[i]),getInnerNodeByIdx(i), sizeof(key_t)*72);
        memcpy(tmp_inner_meta_pool.elementAtIdx(oldToNew[i]),metaOfInnerNodeWithIdx(i), sizeof(Inner_meta));
        Inner_meta *new_meta = tmp_inner_meta_pool.elementAtIdx(oldToNew[i]);
        Inner_meta *old_meta = metaOfInnerNodeWithIdx(i);
        if (old_meta->parent==0){
            assert(i==0);
        }
        else 
            new_meta->parent = tmp_inner_node_pool.elementAtIdx(oldToNew[inner_node_pool.indexAtAddress(old_meta->parent)]);
    }

#pragma omp parallel for
    for (int i=0; i<leaf_num_new;i++) {
        memcpy(tmp_bleaf_node_pool.elementAtIdx(oldToNew_leaf[i]), getBLeafNodeByIdx(i),sizeof(BLeaf_node));
        //TODO: ptr in tmp_bleaf_node_pool is invalid now.     
    }

    normalInnerNodeSet.clear();
    inner_node_pool.reset(tmp_inner_node_pool.getStart(), tmp_inner_node_pool.getEnd());
    inner_meta_pool.reset(tmp_inner_meta_pool.getStart(), tmp_inner_meta_pool.getEnd());
    leaf_pool.reset(tmp_bleaf_node_pool.getStart(), tmp_bleaf_node_pool.getEnd());

    new_levelIdx[0] = 0;
    for (int i=1;i<height;i++) {
        new_levelIdx[i] = new_levelIdx[i-1] + new_levelNum[i-1];
    }

    
    free(levelNum);
    free(levelIdx);
    free(prefixArray);

    levelNum = new_levelNum;
    levelIdx = new_levelIdx;
    prefixArray = new_prefixArray;
    root = getInnerNodeStart(); 
}
void BPlusTree::rebuild_without_leaf(){

    Mem_Pool<Inner_node> tmp_inner_node_pool;
    Mem_Pool<Inner_meta> tmp_inner_meta_pool;
    Mem_Pool<BLeaf_node> tmp_bleaf_node_pool;
    
    int *new_levelNum = (int*)malloc(sizeof(int) * height);
    int *new_levelIdx = (int*)malloc(sizeof(int) * height);
    
    int node_num_new = inner_node_pool.getSize();
    int leaf_num_new = leaf_pool.getSize();
    
    int *new_prefixArray = (int*)malloc(sizeof(int) * node_num_new*2); // x2 for possible grow of prefix, which is not necessary since we use this update method of prefix.

    int *oldToNew = (int *)malloc(sizeof(int) * node_num_new);
    int *oldToNew_leaf = (int *)malloc(sizeof(int) * leaf_num_new); 
    
    for (int i=0;i<height;i++) {
        new_levelNum[i] = 0;
        new_levelIdx[i] = 0;
    }

    int nodeIdx = 0;    //root;
    int inner_size_wo_last_inner_node = getInnerSize_wo_last_inner_node();  //levelIdx[height-2]
    queue<q_ele> inner_node_q;
    inner_node_q.push({nodeIdx,0});             //root as height 0,    para 'height' start from 1.
    
    queue<int> leaf_q;

    
    int pos = -1;
    new_prefixArray[0] = 1;
    while (!inner_node_q.empty()){
  
        pos++;
        
        int nodeIdx = inner_node_q.front().idx;
        int h = inner_node_q.front().height;  
        new_levelNum[h]++;


        Inner_node* n = getInnerNodeByIdx(nodeIdx);
        //cout<<nodeIdx<<" "<<h<<endl;
        
        inner_node_q.pop();

        //memcpy (tmp_inner_node_pool.getElement(),n, sizeof(key_t)*72);
        //memcpy (tmp_inner_meta_pool.getElement(),metaOfInnerNodeWithIdx(nodeIdx), sizeof(Inner_meta));
        oldToNew[nodeIdx] = pos;

        int size = metaOfInnerNodeWithIdx(nodeIdx)->size;   //num of key
        new_prefixArray[pos+1] = new_prefixArray[pos] +size+1; 

        if (isNormalInnerNode(nodeIdx)) {
            if (h==height-2) {      //last inner node

                leaf_q.push(getLeafIdx((BLeaf_node *)n->child[0])); 
            }
            else{
                for (int i=0;i<=size;i++) {
                    inner_node_q.push({getInnerNodeIdx((Inner_node*)n->child[i]), h+1}); 
                }
            }

        }
        else{
            if (h==height-2) {      //last inner node
                leaf_q.push(nodeIdx - inner_size_wo_last_inner_node);
            }
            else{
                for (int i=0;i<=size;i++) {
                    inner_node_q.push({prefixArray[nodeIdx]+i, h+1});
                }
            }
        }
    }
    pos = -1;
    while (!leaf_q.empty()) {
        pos++;
        int nodeIdx = leaf_q.front();
        //memcpy(tmp_bleaf_node_pool.getElement(), getBLeafNodeByIdx(nodeIdx), sizeof(BLeaf_node));
        oldToNew_leaf[nodeIdx] = pos;
        
        leaf_q.pop();
    }
  
    for (int i=0; i<node_num_new;i++) {
        tmp_inner_node_pool.getElement();
        tmp_inner_meta_pool.getElement();
    }

    for (int i=0; i<leaf_num_new;i++) {
        tmp_bleaf_node_pool.getElement();
    }

    //memcpy

    omp_set_num_threads(OMP_Thread_num);
#pragma omp parallel for
    for (int i=0; i<node_num_new;i++) {
        //memcpy (tmp_inner_meta_pool.getElement(),metaOfInnerNodeWithIdx(nodeIdx), sizeof(Inner_meta));
        memcpy(tmp_inner_node_pool.elementAtIdx(oldToNew[i]),getInnerNodeByIdx(i), sizeof(key_t)*72 + sizeof(BLeaf_node*));   // 8 index 64 key and 1child for leaf 
        memcpy(tmp_inner_meta_pool.elementAtIdx(oldToNew[i]),metaOfInnerNodeWithIdx(i), sizeof(Inner_meta));
        Inner_meta *new_meta = tmp_inner_meta_pool.elementAtIdx(oldToNew[i]);
        Inner_meta *old_meta = metaOfInnerNodeWithIdx(i);
        if (old_meta->parent==0){
            assert(i==0);
        }
        else 
            new_meta->parent = tmp_inner_node_pool.elementAtIdx(oldToNew[inner_node_pool.indexAtAddress(old_meta->parent)]);
 
    
    }

//#pragma omp parallel for
//    for (int i=0; i<leaf_num_new;i++) {
//        memcpy(tmp_bleaf_node_pool.elementAtIdx(oldToNew_leaf[i]), getBLeafNodeByIdx(i),sizeof(BLeaf_node));
//    }

    inner_node_pool.reset(tmp_inner_node_pool.getStart(), tmp_inner_node_pool.getEnd());
    inner_meta_pool.reset(tmp_inner_meta_pool.getStart(), tmp_inner_meta_pool.getEnd());

    new_levelIdx[0] = 0;
    for (int i=1;i<height;i++) {
        new_levelIdx[i] = new_levelIdx[i-1] + new_levelNum[i-1];
    }

    
    free(levelNum);
    free(levelIdx);
    free(prefixArray);

    levelNum = new_levelNum;
    levelIdx = new_levelIdx;
    prefixArray = new_prefixArray;
    
    root = getInnerNodeStart(); 
    normalInnerNodeSet.clear();
    
    int inner_size_wo_last_inner_node2 = getInnerSize_wo_last_inner_node();  //levelIdx[height-2]
    for (int i= inner_size_wo_last_inner_node2; i<levelIdx[height-1];i++ ) {
        markAsNormalInnerNode(i);
    }
}
