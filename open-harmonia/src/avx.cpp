#include "bpt.h"
int Inner_node::getChildIdx_avx2_version2(key_t key) {
    /*{{{*/
    //search index 
    //first 4 index
    __m256i vquery = _mm256_set1_epi64x(key);
    __m256i vec = _mm256_set_epi64x(inner_index[0], inner_index[1], inner_index[2], inner_index[3]);
    __m256i vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
    __m256i vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
    __m256i vcmp = _mm256_or_si256(vcmp1, vcmp2);
    int cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key. 
    cmp = cmp & 0x01010101;
    cmp = __builtin_popcount(cmp);  
    int k = cmp; 

    if (k == 4) {
        //last 4 index
        vec = _mm256_set_epi64x(inner_index[4], inner_index[5], inner_index[6], inner_index[7]);
        vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        vcmp = _mm256_or_si256(vcmp1, vcmp2);
        cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key. 
        cmp = cmp & 0x01010101;
        cmp = __builtin_popcount(cmp);  
        k += cmp;
    }

    
    //search key;
    //first 4 keys 
    int start = k * _inner_indexTokey_fanout;
    vec = _mm256_set_epi64x(inner_key[start], inner_key[start+1], inner_key[start+2], inner_key[start+3]);
    vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
    vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
    vcmp = _mm256_or_si256(vcmp1, vcmp2);
    cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key. 
    cmp = cmp & 0x01010101;
    cmp = __builtin_popcount(cmp); 
    k = cmp;
    
    if (k == 4) {
        //last 4 keys 
        vec = _mm256_set_epi64x(inner_key[start+4], inner_key[start+5], inner_key[start+6], inner_key[start+7]);
        vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        vcmp = _mm256_or_si256(vcmp1, vcmp2);
        cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key. 
        cmp = cmp & 0x01010101;
        cmp = __builtin_popcount(cmp); 
        k += cmp;
    }
    return start+k;
/*}}}*/
}



#ifdef TREE_32
void BPlusTree::getChildIdx_avx_4keys(Inner_node **nodes, key_t *keys, int* rets) {
/*{{{*/
    Inner_node *node0=nodes[0];
    Inner_node *node1=nodes[1];
    Inner_node *node2 = nodes[2];
    Inner_node *node3 = nodes[3];
    
    Inner_node *node4 = nodes[4];
    Inner_node *node5 = nodes[5];
    Inner_node *node6 = nodes[6];
    Inner_node *node7 = nodes[7];
    __m256i vquery=_mm256_set_epi32(keys[0],keys[1],keys[2],keys[3],keys[4],keys[5],keys[6],keys[7]);
    __m256i k=_mm256_setzero_si256();
    __m256i one=_mm256_set1_epi32(1);
    __m256i fan=_mm256_set1_epi32(_inner_indexTokey_fanout);
    for(int i=0;i<Idxnum_inner;i++){
        __m256i vec=_mm256_set_epi32(node0->inner_index[i],node1->inner_index[i],node2->inner_index[i],node3->inner_index[i],node4->inner_index[i],node5->inner_index[i],node6->inner_index[i],node7->inner_index[i]);
        __m256i vcmp1=_mm256_cmpgt_epi32(vquery,vec);
        __m256i vcmp2=_mm256_cmpeq_epi32(vquery,vec);
        __m256i vcmp=_mm256_or_si256(vcmp1,vcmp2);

        int cmp=_mm256_movemask_epi8(vcmp);//each 4bit in cmp for a 32-bit key
        if(cmp==0) break;
        __m256i tmp=_mm256_and_si256(vcmp,fan);
        k=_mm256_add_epi32(tmp,k);
    }

   // __m256i tmp=_mm256_set1_epi32(_inner_indexTokey_fanout);
    __m256i start=k;
    
    key_t * inner_key0=&(node0->inner_key[_mm256_extract_epi32(start,7)]);
    key_t * inner_key1=&(node1->inner_key[_mm256_extract_epi32(start,6)]);
    key_t * inner_key2=&(node2->inner_key[_mm256_extract_epi32(start,5)]);
    key_t * inner_key3=&(node3->inner_key[_mm256_extract_epi32(start,4)]);
    
    key_t * inner_key4=&(node4->inner_key[_mm256_extract_epi32(start,3)]);
    key_t * inner_key5=&(node5->inner_key[_mm256_extract_epi32(start,2)]);
    key_t * inner_key6=&(node6->inner_key[_mm256_extract_epi32(start,1)]);
    key_t * inner_key7=&(node7->inner_key[_mm256_extract_epi32(start,0)]);
    
    k=_mm256_setzero_si256();

    for(int i=0;i<_inner_indexTokey_fanout;i++){
        __m256i vec=_mm256_set_epi32(inner_key0[i],inner_key1[i],inner_key2[i],inner_key3[i],inner_key4[i],inner_key5[i],inner_key6[i],inner_key7[i]);
        __m256i vcmp1=_mm256_cmpgt_epi32(vquery,vec);
        __m256i vcmp2=_mm256_cmpeq_epi32(vquery,vec);
        __m256i vcmp=_mm256_or_si256(vcmp1,vcmp2);
        
        int cmp=_mm256_movemask_epi8(vcmp);
        if(cmp==0) break;

        __m256i tmp=_mm256_and_si256(vcmp,one);
        k=_mm256_add_epi32(tmp,k);
    }
    k=_mm256_add_epi32(k,start);
    rets[7]=(int)_mm256_extract_epi32(k,0);
    rets[6]=(int)_mm256_extract_epi32(k,1);
    rets[5]=(int)_mm256_extract_epi32(k,2);
    rets[4]=(int)_mm256_extract_epi32(k,3);
    
    rets[3]=(int)_mm256_extract_epi32(k,4);
    rets[2]=(int)_mm256_extract_epi32(k,5);
    rets[1]=(int)_mm256_extract_epi32(k,6);
    rets[0]=(int)_mm256_extract_epi32(k,7);
/*}}}*/
}

#else 
void BPlusTree::getChildIdx_avx_4keys(Inner_node **nodes, key_t *keys, int* rets) {
/*{{{*/
    Inner_node *node0 = nodes[0];
    Inner_node *node1 = nodes[1];
    Inner_node *node2 = nodes[2];
    Inner_node *node3 = nodes[3];
    

    __m256i vquery = _mm256_set_epi64x(keys[0], keys[1], keys[2], keys[3]);
    __m256i k = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi64x(1);

    //search index 
    for (int i=0;i<Idxnum_inner;i++){
    
        __m256i vec = _mm256_set_epi64x(node0->inner_index[i], node1->inner_index[i], node2->inner_index[i], node3->inner_index[i]);
        __m256i vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        __m256i vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        __m256i vcmp = _mm256_or_si256(vcmp1, vcmp2);

        int cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key.
        if (cmp==0) break;
        
        __m256i tmp = _mm256_and_si256(vcmp, one);
        k = _mm256_add_epi64(tmp, k);

    } 

    __m256i tmp = _mm256_set1_epi64x(_inner_indexTokey_fanout); 
    __m256i start = _mm256_mul_epu32(k, tmp); 
    
    key_t * inner_key0 = &(node0->inner_key[_mm256_extract_epi64(start,3)]);
    key_t * inner_key1 = &(node1->inner_key[_mm256_extract_epi64(start,2)]);
    key_t * inner_key2 = &(node2->inner_key[_mm256_extract_epi64(start,1)]);
    key_t * inner_key3 = &(node3->inner_key[_mm256_extract_epi64(start,0)]);
    // TODO: change these 4 sentence to avx is possible.
    k = _mm256_setzero_si256();

    //search key
    for (int i=0;i<_inner_indexTokey_fanout;i++) {
        __m256i vec = _mm256_set_epi64x(inner_key0[i], inner_key1[i], inner_key2[i], inner_key3[i]);
        __m256i vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        __m256i vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        __m256i vcmp = _mm256_or_si256(vcmp1, vcmp2);

        int cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key.
        if (cmp==0) break;
        
        __m256i tmp = _mm256_and_si256(vcmp, one);
        k = _mm256_add_epi64(tmp, k);
    
    }
    k = _mm256_add_epi64(k, start);
    rets[3] = (int)_mm256_extract_epi64(k,0); 
    rets[2] = (int)_mm256_extract_epi64(k,1); 
    rets[1] = (int)_mm256_extract_epi64(k,2); 
    rets[0] = (int)_mm256_extract_epi64(k,3); 
    

/*}}}*/
}
#endif




#ifdef TREE_32 
void BPlusTree::getChildIdx_avx_4keys_profile(Inner_node **nodes, key_t *keys, int* rets, int &cmp_times_fact, int &cpm_times_ideal) {
    //TODO
}


#else 

void BPlusTree::getChildIdx_avx_4keys_profile(Inner_node **nodes, key_t *keys, int* rets, int &cmp_times_fact, int &cmp_times_ideal) {
/*{{{*/
    Inner_node *node0 = nodes[0];
    Inner_node *node1 = nodes[1];
    Inner_node *node2 = nodes[2];
    Inner_node *node3 = nodes[3];
    

    __m256i vquery = _mm256_set_epi64x(keys[0], keys[1], keys[2], keys[3]);
    __m256i k = _mm256_setzero_si256();
    __m256i one = _mm256_set1_epi64x(1);


    //search index 
    for (int i=0;i<Idxnum_inner;i++){
    
        __m256i vec = _mm256_set_epi64x(node0->inner_index[i], node1->inner_index[i], node2->inner_index[i], node3->inner_index[i]);
        __m256i vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        __m256i vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        __m256i vcmp = _mm256_or_si256(vcmp1, vcmp2);

        int cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key.
        if (cmp==0) {
            break;
        }
        
        __m256i tmp = _mm256_and_si256(vcmp, one);
        k = _mm256_add_epi64(tmp, k);

    } 
    

    int total_cmp_fact = 0;
    int total_cmp_ideal = 0;
    int iddx = -1;
    int big_iddx = -1;
    for (int ii = 0; ii<4;ii++) {
        iddx = _mm256_extract_epi64(k, ii);
        total_cmp_ideal += iddx + 1; //iddx == 0 比较了1次, ==1比较了2次。。。==7比较了8次，(在最后一个循环k++前break掉)， ==8也比较了8次。 这里就不管这么多了..
        if (iddx > big_iddx) {
            big_iddx = iddx;
        }
    }
    total_cmp_fact += (big_iddx+1)*4;
    

    __m256i tmp = _mm256_set1_epi64x(_inner_indexTokey_fanout); 
    __m256i start = _mm256_mul_epu32(k, tmp); 
    
    key_t * inner_key0 = &(node0->inner_key[_mm256_extract_epi64(start,3)]);
    key_t * inner_key1 = &(node1->inner_key[_mm256_extract_epi64(start,2)]);
    key_t * inner_key2 = &(node2->inner_key[_mm256_extract_epi64(start,1)]);
    key_t * inner_key3 = &(node3->inner_key[_mm256_extract_epi64(start,0)]);
    // TODO: change these 4 sentence to avx is possible.
    k = _mm256_setzero_si256();

    
    int count2 = -1;
    
    //search key
    for (int i=0;i<_inner_indexTokey_fanout;i++) {
        __m256i vec = _mm256_set_epi64x(inner_key0[i], inner_key1[i], inner_key2[i], inner_key3[i]);
        __m256i vcmp1 = _mm256_cmpgt_epi64(vquery, vec);
        __m256i vcmp2 = _mm256_cmpeq_epi64(vquery, vec);
        __m256i vcmp = _mm256_or_si256(vcmp1, vcmp2);

        int cmp = _mm256_movemask_epi8(vcmp);// create mask for each 8-bit. so 8 bit in cmp for one 64bit key.
        if (cmp==0)  {
            break;
        }
        __m256i tmp = _mm256_and_si256(vcmp, one);
        k = _mm256_add_epi64(tmp, k);
    
    }


    iddx = -1;
    big_iddx = -1;
    for (int ii = 0; ii<4;ii++) {
        iddx = _mm256_extract_epi64(k, ii);
        total_cmp_ideal += iddx + 1; //iddx == 0 比较了1次, ==1比较了2次。。。==7比较了8次，(在最后一个循环k++前break掉)， ==8也比较了8次。 这里就不管这么多了..
        if (iddx > big_iddx) {
            big_iddx = iddx;
        }
    }
    total_cmp_fact += (big_iddx+1)*4;

    cmp_times_fact = total_cmp_fact;
    cmp_times_ideal = total_cmp_ideal;

    k = _mm256_add_epi64(k, start);
    rets[3] = (int)_mm256_extract_epi64(k,0); 
    rets[2] = (int)_mm256_extract_epi64(k,1); 
    rets[1] = (int)_mm256_extract_epi64(k,2); 
    rets[0] = (int)_mm256_extract_epi64(k,3); 



/*}}}*/
}

#endif

