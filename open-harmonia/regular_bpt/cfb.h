#ifndef CFB_DEF
#define CFB_DEF

#include <climits>
#include "../src/mempool.h"
#include "regular_bpt.h"


//#define key_t long long
//#define Max_Key ((long long)(LLONG_MAX))
//#define value_t long long

#define c_ptr_t int               //child pointer type (offset)
#define r_ptr_t int                //record potiner type (offset)
#define PREFIX_DEFAULT_LENGTH 10000000
typedef union cArr_t{
    c_ptr_t child[DEFAULT_ORDER];
    r_ptr_t record[DEFAULT_ORDER];
    
}cArr_t;    //child arr

typedef struct kArr_t{
    key_t key[DEFAULT_ORDER]; 
}kArr_t;     //key arr


class CFB{

    public: 
        Mem_Pool<cArr_t> *pointer_section;
        Mem_Pool<kArr_t> *key_section;
        Mem_Pool<record> *record_section;
       
        // aux value invalid when upgrade;
        int *prefix;
        int internal_node_num;
        int node_num;
        int height;

        CFB() {
            pointer_section = new Mem_Pool<cArr_t>();
            key_section = new Mem_Pool<kArr_t>();
            record_section = new Mem_Pool<record>();
        }
        
        int getRootIdx() {
            return 0;
        }
        int getHeight() {
            return height;
        }
        
        record * search(key_t key,bool verbose);
        void generateChild(int nodeIdx);
        void generatePointerSection(){
            for (int i=0;i<internal_node_num;i++)
                generateChild(i);
        }
};





CFB *generateCFB(node *root);    //generateCFB according a regular tree



//aux function for generateCFB by a root
void traverse_transform(CFB *cfb, node *node);



class RB {
    
    public: 
        Mem_Pool<cArr_t> *pointer_section;
        Mem_Pool<kArr_t> *key_section;
        Mem_Pool<record> *record_section;
        int node_num;
        int height;
        int rootIdx;
        RB() {
            pointer_section = new Mem_Pool<cArr_t>();
            key_section = new Mem_Pool<kArr_t>();
            record_section = new Mem_Pool<record>();
        }
        int getRootIdx() {
            return rootIdx;
        }
        int getHeight() {
            return height;
        }
};


RB *generateRB(node *root);    //generateRB according a regular tree

//aux function for generateRB by a root
void traverse_transform_RB(RB *rb, node *node);
#endif
