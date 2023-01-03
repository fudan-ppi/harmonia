#include"cfb.h"
#include<iostream>
using namespace std;
record * CFB::search(key_t key,bool verbose){
    int idx = 0;
    int h = height;
    int i=0;

    //find inner_node
    if(verbose) cout<<key;
    while(h>0){
        if(verbose) cout<<"\t"<<idx;
        kArr_t * pkeys = key_section->elementAtIdx(idx);
        i=0;
        while(i<DEFAULT_ORDER-1){
            if(key >= pkeys->key[i]) i++;
            else break;
        }
        if(verbose) cout<<"\t"<<i;
        idx = prefix[idx]+i;
        --h;
    }

    //find leaf
    if(verbose) cout<<"\t"<<idx;
    kArr_t * pkeys = key_section->elementAtIdx(idx);
    record * ret = nullptr;
    for(i=0;i<DEFAULT_ORDER-1;i++){
        if(key == pkeys->key[i]){
            if(verbose) cout<<"\t"<<i; 
            ret = record_section->elementAtIdx(pointer_section->elementAtIdx(idx-internal_node_num)->record[i]);
            break;
        }
    }

    if(verbose && ret==nullptr) cout<<"-1";//not find

    cout<<endl;
    return ret;//not find
}

void CFB::generateChild(int nodeIdx){
    cArr_t *c_arr = pointer_section->elementAtIdx(nodeIdx);
    int childNum = prefix[nodeIdx+1] - prefix[nodeIdx];
    for (int i=0; i<childNum; i++) {
        c_arr->child[i] = i+prefix[nodeIdx];
    }
}
