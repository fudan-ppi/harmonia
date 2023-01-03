
#include"regular_bpt.h"
#include "../src/conf.h"
#include<iostream>



static void *thread_roution(void *args){
    Parallel_Data *data = (Parallel_Data *) args;
    int my_idx = data->thread_idx;
    int offset = data->offset;
    int thread_num = data->thread_num;
    node *root = data->bpt_root;
    std::vector<int> &isInsertArr = *(data->isInsertArr);
    const std::vector<key_t> &keys = *(data->keys);
    const std::vector<val_t> &vals = *(data->keys);
    while(my_idx<data->batch_size){
    
        record * pointer;
	    node * leaf;

        key_t key = keys[my_idx+offset];
        val_t value = vals[my_idx+offset];
        /* The current implementation ignores
        * duplicates.
        */

        //However, since find() use leaf->keys_num, which can be changed in other threads, so even the result of find is NULL, the key may exists in the tree, so we need double check. But, this bug never appears in practice, we do not do the double check in the orignial version in regular_bpt.cpp
        if (find(root, key, false) != NULL){
            isInsertArr[my_idx] = -1;
            my_idx+=thread_num;
            continue;
        }
        

        /* Create a new record for the
         * value.
         */
        pointer = make_record(value);


        /* Case: the tree already exists.
         * (Rest of function body.)
         */

        leaf = find_leaf(root, key, false);

        /* Case: leaf has room for key and pointer.
         */
        pthread_mutex_lock(&leaf->lock);

        //double check
        int i = 0;
        while ((i<leaf->num_keys)&&(leaf->keys[i] <= key)) {
            if (leaf->keys[i] == key) {
                isInsertArr[my_idx] = -1;
                goto out1;
            }
            i++;
        }

        if (leaf->num_keys < order - 1) {
            leaf = insert_into_leaf(leaf, key, pointer);
            pthread_mutex_unlock(&leaf->lock);
            isInsertArr[my_idx] = 1;
            //return root;
        }else {
out1:
            pthread_mutex_unlock(&leaf->lock);
        }
        my_idx+=thread_num;


	/* Case:  leaf must be split.
	 */

	//return insert_into_leaf_after_splitting(root, leaf, key, pointer);
    }
}

node * insert_parallel_batch( node * root,const std::vector<key_t> &keys, int batch_size){
    int queries_num = keys.size();
    int thread_num = CPU_LOGICAL_CORE;
    Parallel_Data data ;
    data.batch_size = batch_size;
    data.keys = &keys;
    data.thread_num = thread_num;

    int dealed_queries_num = 0;

    while (dealed_queries_num < queries_num) {
        
        std::vector<int> isInsertArr(batch_size,0);
        data.isInsertArr = &isInsertArr;
        data.offset = dealed_queries_num;
        data.bpt_root = root;
        
        std::vector<pthread_t> tids;
        std::vector<Parallel_Data> datas(thread_num,data);
        pthread_t tid;
        for(int i=0;i<thread_num;i++){
            datas[i].thread_idx = i;
            if(pthread_create(&tid,NULL,thread_roution,(void*)&datas[i])!=0){
                throw "thread create error";
            }else{
                tids.push_back(tid);
            }
        }
        for(int i=0;i<thread_num;i++){
            pthread_join(tids[i],NULL);
        }

#ifdef ENABLE_TEST
    std::cout << "test para search start"<<std::endl;
    for (int i=0; i<batch_size; i++) {
        int ii = i + dealed_queries_num;
        record * r = find(root, keys[ii], false);
        if (isInsertArr[i]==1) {
            if ((r==NULL) || (r->value!=keys[ii])){
                find(root, keys[ii],true);
                std::cout << keys[ii]<<std::endl;
            }
        }
    }
    std::cout<<" test para end"<<std::endl;
#endif

        for(int i=0;i<batch_size;i++){
            int ii = i+dealed_queries_num;
            if(isInsertArr[i]==1) continue;
            if(isInsertArr[i]==-1){
                //std::cout<<"repeat:"<<keys[i]<<std::endl;
                continue;
            }
            root = insert(root,keys[ii],keys[ii]);
        }
        
        dealed_queries_num += batch_size;
    
    }
   
#ifdef ENABLE_TEST
    std::cout << "test search start"<<std::endl;
    for (int i=0; i<keys.size(); i++) {
        record * r = find(root, keys[i], false); 
        if ((r==NULL) || (r->value!=keys[i])){
            std::cout << keys[i]<<std::endl;
        }
    }
#endif
    
    return root;
}



