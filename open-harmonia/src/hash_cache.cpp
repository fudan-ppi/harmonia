
#ifdef BOOST


#include "hash_cache.h"

/*{{{*/


    int HashCache::lookup(key_t key, value_t &val) {
        //return -1;
        int hash = getHash(key);

        boost::upgrade_mutex *um = &read_write_mutex[hash];
       
        //read_lock rlock(read_write_mutex[hash], boost::try_to_lock);
        //read_lock rlock(read_write_mutex[hash]);
        
        if (um->try_lock_upgrade()) {
        
        //if (rlock.owns_lock()){
            //printf("in\n");
            cache_element *e = &hash_table[hash];
            
            if (e->key==key) {
                val = e->val;
                hits++;   
                
                //return 0;

                
                if (e->existed_time<=cache_threshold2) {
                    um->unlock_upgrade();
                    return 0;
                }
                
                //upgrade_write_lock wlock(rlock);
               
                
                if (um->try_unlock_upgrade_and_lock()) {
                
                    e->existed_time = 0;
                    
                    um->unlock();

                    return 0;
                
                }
                um->unlock_upgrade();
                return 0;

            }
            else{
                //return -1;

                //upgrade_write_lock wlock(rlock);
               
                
                if (um->try_unlock_upgrade_and_lock()) {
                    //printf("upgrade\n");
                    e->existed_time++;
                    um->unlock();
                    return -1;
                }
                
                um->unlock_upgrade();
                return -1;

            }
        
        }
        return -1;

    }

    void HashCache::addCache(key_t key, value_t val) {
        //return ;
        if (content>100) return;

        int hash = getHash(key);
        
        if (hash_table[hash].key==key) return;
        //read_lock rlock(read_write_mutex[hash], boost::try_to_lock);
        
        boost::upgrade_mutex *um = &read_write_mutex[hash];

        if (um->try_lock_upgrade()){
        //if (rlock.owns_lock()) {
            cache_element *e = &hash_table[hash];
            
            if ((e->key!=key)&&(e->existed_time > cache_threshold1)){
            //if (e->key==0){
                //upgrade_write_lock wlock(rlock);
                
                if (um->try_unlock_upgrade_and_lock()) {
                    if (e->key==0) 
                        content++;
                    e->existed_time = 0;
                    e->key = key;
                    e->val = val;
                    um->unlock();
                }
            }
            um->unlock_upgrade();
            return;
        }
        return;

    }

    /*}}}*/


#endif



