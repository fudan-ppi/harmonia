#include "bpt.h"
#include <iostream>
#include <cassert>
#include<string.h>
using namespace std;

//==============================PPI tree create aux 
BLeaf_node * BPlusTree::bleafNodeCreate_PPItree(BLeaf_node *oldnode){
    BLeaf_node *blfnode = NULL; 
    
    blfnode = leaf_pool.getElement();//new slot
    assert(blfnode != NULL);

    //move back
    if(oldnode){
        int oldNode_idx = getLeafIdx(oldnode);
        int moveStart_idx = oldNode_idx + 1;
        bleafnode_MovebackOneSlot(moveStart_idx);
    
        //parent pointer do not need update ,unused 

        blfnode = oldnode+1;
    }

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
Inner_node * BPlusTree::INodeCreate_PPItree(Inner_node *oldnode){
   Inner_node *node = inner_node_pool.getElement();
   assert(node !=NULL);
    //move back
    if(oldnode){
        int oldNode_idx = getInnerNodeIdx(oldnode);

        int moveStart_idx = oldNode_idx + 1;
        

        //parent pointer do not need update 
                //update parent's child info 
         
        Inner_meta *meta =metaOfInnerNodeWithIdx(oldNode_idx);
        Inner_node *parent = meta->parent+1;
        if(parent){
            int prefixArray_idx = getInnerNodeIdx(parent);
            for( ;prefixArray_idx < inner_node_pool.getSize();prefixArray_idx++){
                prefixArray[prefixArray_idx]++;
            }
        }
        // now move back 
        innernode_MovebackOneSlot(moveStart_idx);
        
        //prefix_array move back 
        int *src = prefixArray + moveStart_idx;
        int *dst = src+1;
        size_t s = (inner_node_pool.getSize()-moveStart_idx)*sizeof(int);
        memmove(dst,src,s);

        IMetaCreate_PPItree(oldNode_idx);

        
        node = oldnode+1;
        int index = getInnerNodeIdx(node);

        int level = 0;
        while(index >= levelIdx[level] && level<height-1)  level++;
        for (int i=level;i<height-1;i++) levelIdx[i]++; 
        levelNum[level-1]++;
        
        int update_parent_info_start = prefixArray[moveStart_idx+1]+0;//next node the first child 
        //update meta's parent info 
        if(oldNode_idx < getInnerSize_wo_last_inner_node()){
            meta = metaOfInnerNodeWithIdx(update_parent_info_start);
            for(;meta<inner_meta_pool.getEnd();meta++){
                meta->parent++;
            }
        }

    }


   //Change 0 TO Max_key
   {
        for(int i=0;i<Idxnum_inner;i++) node->inner_index[i] = Max_Key; 
        for(int i=0;i<BPT_ORDER;i++) node->inner_key[i] = Max_Key;
   }
   return node;

}
Inner_meta * BPlusTree::IMetaCreate_PPItree(int old_idx){
    Inner_meta *meta = inner_meta_pool.getElement();
    assert(meta !=NULL);
    inner_meta_pool.moveBackOneSlot(old_idx+1);

    meta = metaOfInnerNodeWithIdx(old_idx)+1;
    meta->index_size = 0;
    meta->size = 0;
    meta->parent = 0;
    return meta;

}

//------------------------------------------------------------ Search aux function 

Inner_node *BPlusTree::search_last_level_inner_node_PPItree(key_t key){
    
    if(height == 1) return NULL;
    int tmp_height = height;
    
    int inner_node_idx = 0;//root;
    Inner_node * node = getInnerNodeByIdx(inner_node_idx);
    
    
    while(tmp_height > 2){
        
        Inner_meta *meta = NULL;
        
        if (key==Max_Key) 
            meta = metaOfInnerNodeWithIdx(inner_node_idx);
        //search index;
        int child_idx = node->getChildIdx(meta,key);
        inner_node_idx = prefixArray[inner_node_idx] + child_idx;

        node = getInnerNodeByIdx(inner_node_idx);
        tmp_height--;  
    }
    return node;//last inner node 
}
value_t BPlusTree::search_leaf_PPItree(Inner_node* last_inner_node,key_t key,int &relist_idx,BLeaf_node *&blfnode){
    blfnode =NULL;
    if(last_inner_node==NULL){ 
        blfnode = (BLeaf_node *)root;
        relist_idx = 0;
    }else {
        if (key==Max_Key) 
            relist_idx = last_inner_node->getChildIdx( metaOfInnerNodeWithAddress(last_inner_node),key);
        else
            relist_idx = last_inner_node->getChildIdx( NULL,key);
        int blfnode_idx  = getInnerNodeIdx(last_inner_node) - getInnerSize_wo_last_inner_node();
        blfnode = getBLeafNodeByIdx(blfnode_idx);
    }
    return blfnode->findKey(relist_idx,key);
}

value_t BPlusTree::searchAndUpdate_leaf_PPItree(Inner_node* last_inner_node,key_t key,int &relist_idx,BLeaf_node *&blfnode, value_t v){
    blfnode =NULL;
    if(last_inner_node==NULL){ 
        blfnode = (BLeaf_node *)root;
        relist_idx = 0;
    }else {
        if (key==Max_Key) 
            relist_idx = last_inner_node->getChildIdx( metaOfInnerNodeWithAddress(last_inner_node),key);
        else
            relist_idx = last_inner_node->getChildIdx( NULL,key);
        int blfnode_idx  = getInnerNodeIdx(last_inner_node) - getInnerSize_wo_last_inner_node();
        blfnode = getBLeafNodeByIdx(blfnode_idx);
    }
    return blfnode->findKeyAndUpdate(relist_idx,key,v);
}





value_t BPlusTree::search_PPItree(key_t key){

    Inner_node *last_inner_node = search_last_level_inner_node_PPItree(key);
    
    int relist_idx;
    BLeaf_node *blfnode = NULL;
    return  search_leaf_PPItree(last_inner_node, key, relist_idx, blfnode);
}

void BPlusTree::updateChildernInfo(Inner_node *node1,Inner_node *node2){
    
    updatePrefixArray(node1,node2);

    int start = prefixArray[getInnerNodeIdx(node1)];
    int child_size = metaOfInnerNodeWithAddress(node1)->size+1;
    for(int i=0;i<child_size;i++){
        metaOfInnerNodeWithAddress(getInnerNodeByIdx(start+i))->parent = node1;
    }

    start = prefixArray[getInnerNodeIdx(node2)];
    child_size = metaOfInnerNodeWithAddress(node2)->size+1;
    for(int i=0;i<child_size;i++){
        metaOfInnerNodeWithAddress(getInnerNodeByIdx(start+i))->parent = node2;
    }
}
void BPlusTree::updatePrefixArray(Inner_node *node1,Inner_node *node2){
    int pre_idx1 = getInnerNodeIdx(node1);
    int pre_idx2 = getInnerNodeIdx(node2);//=pre_idx1+1
    int start = metaOfInnerNodeWithIdx(pre_idx1)->size+1 + prefixArray[pre_idx1];
    prefixArray[pre_idx2] = start;
}

void BPlusTree::check(){
    Inner_node *node = inner_node_pool.getStart();
    for(;node<inner_node_pool.getEnd();node++){
        int node_idx = getInnerNodeIdx(node);
        if(node_idx<getInnerSize_wo_last_inner_node()){
            int child_start = prefixArray[node_idx];
            int child_num = metaOfInnerNodeWithIdx(node_idx)->size+1;
            for(int i=0;i<child_num;i++){
                Inner_node *child = getInnerNodeByIdx(child_start+i);
            
                if(metaOfInnerNodeWithAddress(child)->parent != node ){
                    cout<<"parent: "<< node_idx<<endl;
                    cout<<"child: " << child_start+i<<endl;
                    cout<<"error child infor"<<endl;   
                    exit(0);
                }
            }
        }
        else{
        //last_inner_node
        }
    }
}
