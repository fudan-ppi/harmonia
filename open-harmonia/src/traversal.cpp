#include "bpt.h"
#include <cstring>
using namespace std;
void travese_print(Inner_node *node, Inner_meta *meta , int height) {
    

    
    cout<<"h = "<<height<<";  index_size = "<<meta->index_size<<";   key_size = "<<meta->size<<"; min_key: "<<node->inner_key[0]<<"; max_key: "<<node->inner_key[meta->size-1]<<endl; 
    
}
void BPlusTree::traverse(void *node, int level) {
    if (leaf_pool.inPool(node)) {
   
        return;
    }
    
    assert(inner_node_pool.inPool(node));
    Inner_node *i_node = (Inner_node *)node;

    Inner_meta *meta = metaOfInnerNodeWithAddress(i_node);
    travese_print(i_node, meta, level);   
    for (int i=0; i<=meta->size;i++) {
       traverse(i_node->child[i], level+1); 
    }

}

template <typename element_type>
static inline element_type *genNewPtr(element_type *old, element_type *old_start, element_type *new_start, int* oldToNew) {
    if (old==NULL) return NULL;
    return new_start + oldToNew[(old - old_start)];
}


void BPlusTree::sort(BPlusTree *&sortedTree) {
/*{{{*/
    oldToNew = (int *)malloc(inner_node_pool.getSize() * sizeof(int));
    newToOld = (int *)malloc(inner_node_pool.getSize() * sizeof(int));
    levelNum = (int *)malloc(height * sizeof(int));
    levelIdx = (int *)malloc(height * sizeof(int));
    int* localLevelIdx = (int *)malloc(height * sizeof(int));


    oldToNew_leaf = (int *)malloc(leaf_pool.getSize()*sizeof(int));
    newToOld_leaf = (int *)malloc(leaf_pool.getSize()*sizeof(int));
    

    for (int i=0;i<height;i++) {
        levelNum[i] = 0;
        levelIdx[i] = 0;
        localLevelIdx[i] = 0;
    }
    


    traverse_count(root, 0);
    
    levelIdx[0] = 0;
    for (int i=1;i<height;i++) {
        levelIdx[i] = levelIdx[i-1] + levelNum[i-1];
    }

    
    traverse_sort(root, 0, localLevelIdx);

    sortedTree = new BPlusTree(true);

    sortedTree->height = height;
    sortedTree->levelIdx = levelIdx;
    sortedTree->levelNum = levelNum;
    sortedTree->levelIdx = levelIdx;
    sortedTree->oldToNew = oldToNew;
    sortedTree->newToOld = newToOld;
    sortedTree->oldToNew_leaf = oldToNew_leaf;
    sortedTree->newToOld_leaf = newToOld_leaf;
    sortedTree->size = size;

    int inner_node_num = inner_node_pool.getSize();
    Inner_node *old_start_inode = inner_node_pool.getStart(); 
    Inner_node *new_start_inode = sortedTree->inner_node_pool.getStart();

    int leaf_node_num = leaf_pool.getSize();
    BLeaf_node *old_start_bnode = leaf_pool.getStart();
    BLeaf_node *new_start_bnode = sortedTree->leaf_pool.getStart();
    
    
    assert(inner_node_pool.inPool(root));
    sortedTree->root = genNewPtr<Inner_node>((Inner_node *)root, old_start_inode, new_start_inode, oldToNew);


    //inner_meta_pool
    //inner_node_pool
    for (int i=0;i<inner_node_num;i++) {
        Inner_node *new_node = sortedTree->inner_node_pool.getElement(); 
        memcpy(new_node, inner_node_pool.elementAtIdx(newToOld[i]), sizeof(Inner_node));
        Inner_meta *new_meta = sortedTree->inner_meta_pool.getElement();
        memcpy(new_meta, inner_meta_pool.elementAtIdx(newToOld[i]), sizeof(Inner_meta));

        //last inner node    root: Height=0;  leaf_node: Height = height-1;  last_inner_node:  Height = height-2 
        
        
        new_meta->parent = genNewPtr<Inner_node>(new_meta->parent, old_start_inode, new_start_inode, oldToNew);
        if (i<levelIdx[height-2]) { //before last_inner_node 
            for (int j=0;j<new_meta->size+1;j++) {
                Inner_node *old_ptr = (Inner_node *)new_node->child[j];
                new_node->child[j] = genNewPtr<Inner_node>(old_ptr, old_start_inode,new_start_inode,oldToNew);
            }
        }
        else {
            BLeaf_node *old_ptr = (BLeaf_node *)new_node->child[0];
            for (int j=0;j<new_meta->size+1;j++) {
                new_node->child[j] = genNewPtr<BLeaf_node>(old_ptr, old_start_bnode, new_start_bnode, oldToNew_leaf);
            }
        }
    }
    //leaf_pool
    for (int i=0;i<leaf_node_num;i++) {
       BLeaf_node *new_node = sortedTree->leaf_pool.getElement();
       memcpy(new_node, leaf_pool.elementAtIdx(newToOld_leaf[i]),sizeof(BLeaf_node));
       new_node->next = genNewPtr<BLeaf_node>(new_node->next, old_start_bnode, new_start_bnode, oldToNew_leaf);
       new_node->parent = genNewPtr<Inner_node>(new_node->parent, old_start_inode, new_start_inode, oldToNew);
    }
   


    sortedTree->prepareArray();
    //cout<<levelNum[height-2]<<endl;
/*}}}*/
}



void BPlusTree::traverse_count(void *node, int level) {
    if (leaf_pool.inPool(node)) {
   
        return;
    }
    
    assert(inner_node_pool.inPool(node));
    Inner_node *i_node = (Inner_node *)node;
    Inner_meta *meta = metaOfInnerNodeWithAddress(i_node);

    levelNum[level]++;    

    for (int i=0; i<=meta->size;i++) {
        traverse_count(i_node->child[i], level+1); 
    }

}

void BPlusTree::traverse_sort(void *node, int level, int *localLevelIdx) {
    if (leaf_pool.inPool(node)) {
        
        BLeaf_node *b_node = (BLeaf_node *)node;
        oldToNew_leaf[leaf_pool.indexAtAddress(b_node)] = localLevelIdx[level];
        newToOld_leaf[localLevelIdx[level]] = leaf_pool.indexAtAddress(b_node);
        localLevelIdx[level]++;
        return;
    }
    
    assert(inner_node_pool.inPool(node));
    Inner_node *i_node = (Inner_node *)node;
    Inner_meta *meta = metaOfInnerNodeWithAddress(i_node);

    
    oldToNew[inner_node_pool.indexAtAddress(i_node)] = levelIdx[level] + localLevelIdx[level];
    newToOld[levelIdx[level] + localLevelIdx[level]] = inner_node_pool.indexAtAddress(i_node);
    localLevelIdx[level]++;

    if (level == height-2)  //last_inner_node 
        traverse_sort(i_node->child[0], level+1, localLevelIdx); 
    else for (int i=0; i<=meta->size;i++) {
        traverse_sort(i_node->child[i], level+1, localLevelIdx); 
    }

}



void BPlusTree::prepareArray() {



    
    prefixArray = (int *)malloc(inner_node_pool.getSize() * sizeof(int)*2);
    prefixArray[0] = 1;
   
    //actually the last_inner_node's prefixArray is not necessary
    for (int i = 1;i<inner_node_pool.getSize(); i++) {
        int lastNodeSize = inner_meta_pool.elementAtIdx(i-1)->size + 1; // size 为 Key的size, 所以要+1  
        prefixArray[i] = prefixArray[i-1] + lastNodeSize;
        
    }

}


void BPlusTree::getSplitPrefixArray(unsigned short *&splitPrefixArray, unsigned short *&base){
    

    int size = inner_node_pool.getSize();   //inner node size
    
    base = (unsigned short *)malloc(size/BASE_GRP * sizeof(unsigned short));
    splitPrefixArray = (unsigned short *)malloc(size * sizeof(unsigned short)); 
     
    for (int i=0; i < size; i++) {
        if (i%BASE_GRP==0) {
           base[i/BASE_GRP] = prefixArray[i]/BASE_COEF; 
        }
        splitPrefixArray[i] = prefixArray[i] - base[i/BASE_GRP]*BASE_COEF;

    }

}







//range大小num-1
void BPlusTree::getKeyRangeOfInnerNode(int level, int num, key_t* &range) {

    for (int i=1;i<num;i++) {
        int idx = levelIdx[level] + levelNum[level]/num * i;
        range[i-1] = inner_node_pool.elementAtIdx(idx)->inner_key[0]; 
    }

}





void BPlusTree::empty_cal() {

    int result[8];

    for (int i=0;i<8;i++){
        result[i] = 0;
    }
    int size = inner_node_pool.getSize();
    for (int i=0;i<size;i++){  
        key_t *index = inner_node_pool.elementAtIdx(i)->inner_index;
        int j = 0;
        while ((j<Idxnum_inner)&&(index[j]!=Max_Key)) {
            j++;
        }
        if (j==0) {
            continue;
        }
        result[j-1]++; 
    }  
    for (int i=0;i<8;i++)
        cout<<"node with index length = "<<i<<" : "<<result[i]<<endl;
}
