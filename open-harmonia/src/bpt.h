#ifndef BPT_H
#define BPT_H
#include<stdio.h>
#include<iostream>
#include<vector>
#include<climits>
#include<immintrin.h>
#include "mempool.h"
#include <algorithm>
#include <pthread.h>
#include <unordered_map>
#include <unordered_set>

#include "conf_for_bpt.h"
#define BPT_ORDER 64

#ifdef TREE_32

#define key_t int 
#define Max_Key ((int)(INT_MAX))
#define value_t int

#else 

#define key_t long long
#define Max_Key ((long long)(LLONG_MAX))
#define value_t long long

#endif 
#ifdef TREE_32 

static const char* TYPE_D="%d";  
#else

static const char* TYPE_D="%lld";  

#endif


#define P 16


//for split prefix
#define BASE_COEF (1<<15)
#define BASE_GRP 512  // MAX_2BYTE:2^16    (2^15/fanout_64)      一个group里的值减去对应的base*2^15，一定能用16个bit表示



//magic num 
const int L_Fanout = 4;//leafnode fanout
const int BL_leafnum = 64; // how many leafnode in Bigleaf 

const int _inner_indexTokey_fanout = 8;//1 index to 8 key
const int Idxnum_inner = BPT_ORDER/ _inner_indexTokey_fanout;// how many index in index fragement 

struct Inner_node;
class BPlusTree;
typedef struct{
    const std::vector<key_t> *keys;
    const std::vector<value_t> *vals;
    std::vector<int> *isInsertArr;//0 not update, -1 already has , 1 update
    int thread_idx;
    int thread_num;
    int batch_size;
    BPlusTree *bpt;
    int offset;                     //for two level batch, offset of this batch in keys array.
}Parallel_update_data;

typedef struct{
    int index_size;//how many index in inner_node;max = 7
    int size;// how many key in inner node;max = 63
    Inner_node * parent;//inner_node's parent is Inner_node. This is certainty;

}Inner_meta;

typedef struct Inner_node{
    
    key_t inner_index[Idxnum_inner];//index for inner_key,I[s] = k[8s];
    key_t inner_key[BPT_ORDER];//key of inner node. key[64] = Max long int;
    void *child[BPT_ORDER];//all is same
#ifdef AVX_ALL  
    int getChildIdx(Inner_meta *meta,key_t key){
        //return getChildIdx_avx2(meta, key);
        return getChildIdx_avx2_version2(key);
    }
#else 
    int getChildIdx(Inner_meta *meta,key_t key){
       //search index /*{{{*/
        int i=0;

        if (key==Max_Key) {
            int index_size = meta->index_size;
       
            for(;i < index_size;i++){
                if(key < inner_index[i]) break;
            }
        }
        else{
            //7 = 64/8-1
            for(;i < Idxnum_inner-1;i++){
                if(key < inner_index[i]) break;
            }
        }


        int start = i * _inner_indexTokey_fanout;
        //int end = start + std::min(_inner_indexTokey_fanout,meta->size-start);//min too slow!!!
      
        if (key==Max_Key) {
            int end = start + (_inner_indexTokey_fanout > meta->size-start ? meta->size-start:_inner_indexTokey_fanout);
            
            for(i=start;i<end;i++){
                if(key < inner_key[i]) break; 
            }
        }
        else {
            for(i=start;i<start + 8;i++){
                if(key  < inner_key[i]) break; 
            }
        }
        /*
        for(i=start;i<end;i++){
           if(key < inner_key[i]) break; 
        }*/

        return i;/*}}}*/
    };
#endif 
    int  getChildIdx_avx2(Inner_meta *meta,key_t key){
        //search index/*{{{*/
        //first 4 index
        __m256d Vquery = _mm256_set1_pd((double)key); 
        __m256d vec = _mm256_set_pd((double)inner_index[0],(double)inner_index[1],(double)inner_index[2],(double)inner_index[3]);
       // __m256i Vcmp = _mm256_cmpgt_epi64(Vquery,vec);
        __m256d Vcmp = _mm256_cmp_pd(Vquery,vec,29);

        int cmp = _mm256_movemask_pd(Vcmp);
        //cmp = cmp & 0x10101010;
        cmp = __builtin_popcount(cmp);
        
        int k = cmp;
        //last 4 index
        vec = _mm256_set_pd((double)inner_index[4],(double)inner_index[5],(double)inner_index[6],(double)inner_index[7]);
        //Vcmp = _mm256_cmpgt_epi64(Vquery,vec);
        Vcmp = _mm256_cmp_pd(Vquery,vec,29);
        cmp = _mm256_movemask_pd(Vcmp);
        //cmp = cmp & 0x10101010;
        cmp = __builtin_popcount(cmp);
        k+= cmp; 
        //search key;
        //first 4 key
        int start = k * _inner_indexTokey_fanout;
        
        vec = _mm256_set_pd((double)inner_key[start],(double)inner_key[start+1],(double)inner_key[start+2],(double)inner_key[start+3]);
        //Vcmp = _mm256_cmpgt_epi64(Vquery,vec);
        Vcmp = _mm256_cmp_pd(Vquery,vec,29);
        cmp = _mm256_movemask_pd(Vcmp);
        //cmp = cmp & 0x10101010;
        cmp = __builtin_popcount(cmp);

        k = cmp; 
        //last 4 key
        vec = _mm256_set_pd((double)inner_key[start+4],(double)inner_key[start+5],(double)inner_key[start+6],(double)inner_key[start+7]);
        //Vcmp = _mm256_cmpgt_epi64(Vquery,vec);
        Vcmp = _mm256_cmp_pd(Vquery,vec,29);
        cmp = _mm256_movemask_pd(Vcmp);
        //cmp = cmp & 0x10101010;
        cmp = __builtin_popcount(cmp);

        k+=cmp;
        return start+k;/*}}}*/
    }
    int  getChildIdx_avx2_version2(key_t key);
    
    void * getChild(Inner_meta *meta,key_t key){
        int idx = getChildIdx(meta,key);
        return child[idx];
    };
    void * getChild_avx2(Inner_meta *meta,key_t key){
        int idx = getChildIdx_avx2(meta,key);
        return child[idx];
    };
    void setChild(void *kid,int idx){//for bigleaf
         child[idx] = kid;
    }
    void moveKeyBackward(int idx,Inner_meta *meta){//move idx element to idx+1
        inner_key[idx+1] = inner_key[idx];
        child[idx+2] = child[idx+1];
        if((idx+1)>=_inner_indexTokey_fanout && (idx+1)%_inner_indexTokey_fanout == 0){
            inner_index[(idx+1) / _inner_indexTokey_fanout -1] = inner_key[idx+1];
            meta->index_size = std::max(meta->index_size,(idx+1)/ _inner_indexTokey_fanout);
        }
    }
    void putKeyAndChild(key_t key,void *kid,int idx ,Inner_meta *meta){ 
        inner_key[idx] = key;
        child[idx+1] = kid;
        if(idx>=_inner_indexTokey_fanout && idx%_inner_indexTokey_fanout == 0){
            inner_index[idx/ _inner_indexTokey_fanout -1] = key;
            meta->index_size = std::max(meta->index_size,idx/ _inner_indexTokey_fanout);
        }
    }
    
}Inner_node;


typedef struct{
    key_t r_key;
    value_t  val;
}Record;

inline Record make_record(key_t key,value_t val){
    Record re;
    re.r_key = key;
    re.val =val;
    return re;
}

typedef struct Record_list{
    Record r[L_Fanout];//every recode list hold 4 record in one cache line;

    Record_list(){
        r[0] = make_record(Max_Key,0);
        r[1] = make_record(Max_Key,0);
        r[2] = make_record(Max_Key,0);
        r[3] = make_record(Max_Key,0);
    }
    key_t keyAtIndex(int idx){
        return r[idx].r_key;
    }
    value_t valAtIndex(int idx){
        return r[idx].val;
    }
    void putReordAtIndex(int idx,Record record){
        r[idx] = record;
    }
    Record getReordAtIndex(int idx){
        return r[idx];
    }
}Record_list;

typedef struct BLeaf_node{
    Record_list relist[BL_leafnum];//64
    BLeaf_node *next;
    Inner_node *parent;
    int size[BL_leafnum];
    int used_relist_slot_num;//BL_leafnum
    pthread_mutex_t lock;

    int bleaf_has_slot(){
        return used_relist_slot_num >= BL_leafnum?-1 : used_relist_slot_num;
    }
    value_t findKey(int relist_idx,key_t key){
        /* 
        int i = size[relist_idx]-1;
        Record_list rl = relist[relist_idx];
        Record record;
        for(;i>=0;--i){
            record = rl.getReordAtIndex(i);
            if(key ==record.r_key) return record.val;
        }
        return -1;//not found
        */
        int i = size[relist_idx]-1;
        Record_list *rl = &(relist[relist_idx]);
        Record *record;
        for(;i>=0;--i){
            record = &(rl->r[i]);
            if(key ==record->r_key) {
                return record->val;
            }
        }
        return -1;//not found
        
    }
    value_t findKeyAndUpdate(int relist_idx, key_t key, value_t v){
        int i = size[relist_idx]-1;
        Record_list *rl = &(relist[relist_idx]);
        Record *record;
        for(;i>=0;--i){
            record = &(rl->r[i]);
            if(key ==record->r_key) {
                value_t r = record->val;
                record->val = v;
                return r;
            }
        }
        return -1;//not found
    }
    int findKeyLowerBound(int relist_idx,key_t key){
        int i = 0;
        for(;i<size[relist_idx];i++){
            //std::cout<<"key="<<key<<" relist["<<relist_idx<<"].keyAtIndex("<<i<<")="<<relist[relist_idx].keyAtIndex(i)<<std::endl;
            if(key <= relist[relist_idx].keyAtIndex(i)) break;
        }
        if (i==size[relist_idx]) i=-1;
        return i;
    }
    int findKeyUpperBound(int relist_idx,key_t key){
        int i = size[relist_idx]-1;
        for(;i>=0;i--){

            //std::cout<<"key="<<key<<" relist["<<relist_idx<<"].keyAtIndex("<<i<<")="<<relist[relist_idx].keyAtIndex(i)<<std::endl;
            if(key >= relist[relist_idx].keyAtIndex(i)) break;
        }
        return i;
    }

    value_t getValue(int relist_idx, int index) {
        return relist[relist_idx].valAtIndex(index);
    }
    void getKeyValues(int relist_idx, int start_index, int end_index, std::vector<Record> *vector) {
        
        if (end_index==-1) end_index = size[relist_idx]-1;
        Record *r = relist[relist_idx].r;
        for (int i=start_index; i<=end_index; i++) {
            vector->push_back(r[i]);
        }
    }
    void getKeyValuesRange(int relist_idx_start, int relist_idx_end, int start_index, int end_index, std::vector<Record> *vector) {
        if (relist_idx_end == -1) relist_idx_end = used_relist_slot_num-1;
        if (relist_idx_start>relist_idx_end) return;
        if (relist_idx_start==relist_idx_end) {
            getKeyValues(relist_idx_start, start_index, end_index, vector);
            return;
        }
        getKeyValues(relist_idx_start, start_index, -1, vector);
        
        for (relist_idx_start+=1; relist_idx_start < relist_idx_end; relist_idx_start++) {
            getKeyValues(relist_idx_start, 0, -1, vector); 
        }
        getKeyValues(relist_idx_end, 0, end_index, vector);
    }

    ~BLeaf_node(){
        pthread_mutex_destroy(&lock);
    }
}BLeaf_node;



//---------------------------------------------------------------------
class BPlusTree{
    private:
        int size;                   //how many unique record in BPlusTree 
        int height;                 //B+ tree height (root is level 1);
        Mem_Pool<Inner_node> inner_node_pool;
        Mem_Pool<Inner_meta> inner_meta_pool;
        Mem_Pool<BLeaf_node> leaf_pool;

        
        int *oldToNew;              //hash table:  oldId to newId
        int *newToOld;              //hash table:  newId to oldId
        
        int *levelNum;      //root is level 0; how many nodes in each level             
        int *levelIdx;      //levelIdx[x] = start idx in sorted_pool of level x

        int *oldToNew_leaf;
        int *newToOld_leaf;

        
        int *prefixArray;   //node的size求前项和
        //对于某一个pool位置pos的node来说，child[x]的位置为
        // inner_node_pool[prefixArray[pos] + x]
        //对于last_inner_node来说,child[x]==child[0]的位置位置为
        // leaf_pool[pos- inner_node_size_wo_last_inner_node]

        void *root;

        //create aux function
        Inner_node * INodeCreate();

        Inner_meta * IMetaCreate();
        
        BLeaf_node * bleafNodeCreate();
        BLeaf_node * bleafNodeCreate(key_t key,value_t val);

        //for ppi tree update
        Inner_node *INodeCreate_PPItree(Inner_node *oldnode);
        Inner_meta *IMetaCreate_PPItree(int idx);
        BLeaf_node * bleafNodeCreate_PPItree(BLeaf_node *oldnode);

    public:
        
        bool isPPItree;
        std::unordered_set<int> normalInnerNodeSet;// normal node use child pointer to fetch child
        
        bool isNormalInnerNode(int idx){
            return normalInnerNodeSet.find(idx)!=normalInnerNodeSet.end();
        }

        void  markAsNormalInnerNode(int idx){
            normalInnerNodeSet.insert(idx);
        }
        void changeToNormalInnerNode(int idx);
        
        //search aux function 
        Inner_node *search_last_level_inner_node(key_t key);
        value_t search_leaf(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode);
        value_t searchAndUpdate_leaf(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode, value_t v);
        
        int search_leaf_fuzzy(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode, int boundUpOrLow);

        //search aux function for PPItree :original completely move update 
        Inner_node *search_last_level_inner_node_PPItree(key_t key);
        value_t search_leaf_PPItree(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode);
        value_t searchAndUpdate_leaf_PPItree(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode, value_t v);

        //search aux function for PPITree :new update method 6.22
        Inner_node *search_last_level_inner_node_PPItree_Newupdate(key_t key); 
        value_t search_leaf_PPItree_Newupdate(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode);
        value_t searchAndUpdate_leaf_PPItree_Newupdate(Inner_node* last_inner_node,key_t key,int &leaf_idx,BLeaf_node *&blnode, value_t v);




        //insert aux function 
        void insert_inner_no_split(Inner_node *inode,key_t newkey,void * nextLevelnode);
        void insert_parent(Inner_node *parent,key_t newkey,void *oldNextLevelnode,void *newNextLevelnode);
        void insert_parent_Newupdate(Inner_node *parent,key_t newkey,void *oldNextLevelnode,void *newNextLevelnode);

        void insert_leaf_no_split(BLeaf_node *blnode,key_t key,value_t val,int relist_idx);
    
        //split funtion
        key_t bleaf_inner_split(BLeaf_node* blfnode,key_t key,value_t val,const int relist_idx);// return new key

        key_t bleaf_split(BLeaf_node* blfnode,BLeaf_node*& newblfnode,key_t key,value_t val,int  relist_idx);// return new key
 
        key_t bleaf_split_core(BLeaf_node* blfnode,BLeaf_node*& newblfnode,key_t key,value_t val,int  relist_idx);// return new key

        key_t bleaf_split_Newupdate(BLeaf_node*& blfnode,BLeaf_node*& newblfnode,key_t key,value_t val,int  relist_idx);// return new key
       
        key_t inner_node_split_core(Inner_node *inode,Inner_node*& newInode,key_t key,void *oldNextLevelnode,void *nextLevelnode,bool isNormalInnerNode);// return new key
        key_t inner_node_split(Inner_node *inode,Inner_node*& newInode,key_t key,void *oldNextLevelnode,void *nextLevelnode);// return new key

        //other aux
        Inner_meta *metaOfInnerNodeWithAddress(Inner_node *node){
            int inner_node_idx = inner_node_pool.indexAtAddress(node);
            return inner_meta_pool.elementAtIdx(inner_node_idx);
        }
        Inner_meta *metaOfInnerNodeWithIdx(int idx){
            return inner_meta_pool.elementAtIdx(idx);
        }
        bool check_inner_full(Inner_node* inode){
            return metaOfInnerNodeWithAddress(inode)->size == BPT_ORDER-1 ? true:false;
        }
        bool check_leaf_full(BLeaf_node *blnode,int relist_idx){
            return blnode->size[relist_idx]== L_Fanout? true:false;
        };


        void traverse_count(void *node, int level); 
        void traverse_sort(void *node, int level, int *localLevelIdx);
        void prepareArray();
        void updatePrefixArray(Inner_node *oldnode,Inner_node* newnode);
        void updateChildernInfo(Inner_node *oldnode,Inner_node* newnode);
    public:
        BPlusTree():isPPItree(false),size(0),height(0),root(NULL){
        
        };
        BPlusTree(bool isPPI):isPPItree(isPPI),size(0),height(0),root(NULL){
        };

        void *getRoot(){return root;};
        bool insert(key_t key, value_t val);
        bool insert_Newupdate(key_t key, value_t val);

        //for parallel update;
        bool insert_parallel(const std::vector<key_t> &keys,const std::vector<value_t> &vals);
        bool insert_parallel_Newupdate(const std::vector<key_t> &keys,const std::vector<value_t> &vals);
        bool insert_parallel_Newupdate_batch(const std::vector<key_t> &keys,const std::vector<value_t> &vals, int syn_batch_size, int rebuild_batch_size);
        bool insert_parallel_batch(const std::vector<key_t> &keys,const std::vector<value_t> &vals, int batch_size);

        void rebuild();
        void rebuild_without_leaf();

        value_t search(key_t key);
        value_t search_PPItree(key_t key);
        value_t search_PPItree_Newupdate(key_t key);

        void search_swp(const std::vector<key_t> &keys, std::vector<value_t> &vals);
        void range_search(key_t start_key,key_t end_key, std::vector<Record> *v);
        bool remove(key_t key);
        void getInnerSegementationInfo(Inner_node *&innode,unsigned int &innodeSeg_size_byte){
            inner_node_pool.getMetaForGPU(innode,innodeSeg_size_byte);
        }
        int getHeight(){ return height;};
        int getRootIdx(){
            if(root && inner_node_pool.inPool(root)){
                return inner_node_pool.indexAtAddress((Inner_node *) root);

            }
            return -1;
        }

        void traverse(void *node, int level=1);
        void sort(BPlusTree *&sortedTree);
        Inner_node* getInnerNodeByIdx(int idx){
            return inner_node_pool.elementAtIdx(idx);
        }
        BLeaf_node* getBLeafNodeByIdx(int idx){
            return leaf_pool.elementAtIdx(idx);
        }
        int *getPrefixArray() {
            return prefixArray;
        }
        void getSplitPrefixArray(unsigned short *&splitPrefixArray, unsigned short *&base);
        int getInnerSize() {
            return inner_node_pool.getSize();
        }
        int getBLeafSize() {
            return leaf_pool.getSize();
        }
        BLeaf_node *getLeafByIdx(int idx) {
            return leaf_pool.elementAtIdx(idx);
        }
        int getLeafIdx(BLeaf_node *node){
            return leaf_pool.indexAtAddress(node);
        }
        int getInnerSize_wo_last_inner_node() {
            return levelIdx[height-2];
        }
        Inner_node* getInnerNodeStart() {
            return inner_node_pool.getStart();
        }
        int getInnerNodeIdx(Inner_node *node){
            return inner_node_pool.indexAtAddress(node);
        }
        BLeaf_node* getBLeafNodeStart() {
            return leaf_pool.getStart();
        }
        void getKeyRangeOfInnerNode(int level, int num, key_t *&range);

        void print_LevelNum() {
            for (int i=0;i<height;i++)
                std::cout<<"level "<<i<<" nodes num "<<levelNum[i]<<std::endl;
        }

        void empty_cal();   //empty calculate
        
        int getLevelIdx(int level) {
            return levelIdx[level];
        }
        //for update 
        void bleafnode_MovebackOneSlot(int idx){
            leaf_pool.moveBackOneSlot(idx);
        }
        void innernode_MovebackOneSlot(int idx){
            inner_node_pool.moveBackOneSlot(idx);
        }
        void check();


//--------------AVX OPT-------------------------------

    void getChildIdx_avx_4keys(Inner_node **nodes, key_t *keys, int* rets);
    //read 4 keys in using pointer "keys", write the answer into the given address "rets".

    void getChildIdx_avx_4keys_profile(Inner_node **nodes, key_t *keys, int* rets, int &cmp_times_fact, int &cmp_times_ideal);


};

#endif //BPT_H

