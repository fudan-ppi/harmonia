
#ifdef BOOST



#undef P
#include<boost/thread/locks.hpp>
#include<boost/thread/shared_mutex.hpp>
#include <unordered_map>

#include "bpt.h"


#include <pthread.h>

typedef struct{
    key_t key;
    value_t val;
    int existed_time = 201;
}cache_element;

typedef boost::upgrade_lock<boost::shared_mutex> read_lock;
typedef boost::upgrade_to_unique_lock<boost::shared_mutex> upgrade_write_lock;

class HashCache{
    /*{{{*/

    public:
        //std::unordered_map<key_t, value_t> cache_map;
        //volatile bool access_flag = false; //是否被访问
        int hits;
        int content;

        static const int cache_length = 200;
        static const int cache_threshold1 = 200;    //exist_time到达这个值，addCache会更新
        static const int cache_threshold2 = 190;    //exist_time到达这个值，命中的会更新exist_time为0 
        
        //pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;


        //boost::shared_mutex read_write_mutex[cache_length];
        
        boost::upgrade_mutex read_write_mutex[cache_length];

        cache_element  hash_table[200];

        bool element_flag[200];

        int getHash(key_t key) {
            return (key / 18446744073709551)%200;
        }
        int lookup(key_t key, value_t &val);
        void addCache(key_t key, value_t val);
/*
        int lookup(key_t key, value_t &val) {
            int hash = getHash(key);
           
            read_lock rlock(read_write_mutex[hash], boost::try_to_lock);
            if (rlock.owns_lock()){
                cache_element *e = &hash_table[hash];
                
                if (e->key==key) {
                    val = e->val;
                    hits++;   
                    
                    if (e->existed_time<=cache_threshold2) return 0;
                    
                    upgrade_write_lock wlock(rlock);
                    e->existed_time = 0;
                    return 0;
                }
                else{
                    upgrade_write_lock wlock(rlock);
                    e->existed_time++;
                    return -1;
                }
            
            }

            return -1;
        }

        void addCache(key_t key, value_t val) {
            if (content>100) return;

            int hash = getHash(key);
            
            read_lock rlock(read_write_mutex[hash], boost::try_to_lock);
           
            if (rlock.owns_lock()) {
                cache_element *e = &hash_table[hash];

                if (e->existed_time > cache_threshold1){
                    upgrade_write_lock wlock(rlock);
                    e->existed_time = 0;
                    e->key = key;
                    e->val = val;
                }
                return;
            }
            return;
        }
*/        
        
/*
        int lookup(key_t key, value_t &val){

            //while (1)
            if (access_flag == false) {
                //while (1)
                if (__sync_bool_compare_and_swap(&access_flag, false, true)){
                    //std::cout<<"get!"<<std::endl;
                    //printf("search! %lld\n",key);
                    auto got = cache_map.find(key);
                    access_flag = false;

                    if (got == cache_map.end())
                        return -1;
                    //printf("hit! %lld, %lld\n", got->first,got->second);
                    val = got->second;
                    hits++;
                    return 0;
                }
            }
            return -1;
        }
        void addCache(key_t key, value_t val) {
            
            if (cache_map.size()>200) return;
            
            //while(1)
            if (access_flag == false) {
                //while (1)
                if (__sync_bool_compare_and_swap(&access_flag, false, true)) {
                    auto got = cache_map.find(key);
                    access_flag = false; 
                    
                    if(got != cache_map.end()) return;

                    //由于对加入cache没有那么高的要求，因此完全可以只要没获得锁就放弃这次加入
                    if (__sync_bool_compare_and_swap(&access_flag, false, true)) {
                        cache_map[key]  = val; 
                        access_flag = false;  
                    }
                
                }
            }
            return; 
        }
*/

/*
        int lookup(key_t key, value_t &val) {

            if (access_flag == false) return -1;
            if (pthread_rwlock_tryrdlock(&rwlock)!=0) {return -1;}     //未申请到
            auto got = cache_map.find(key);
            pthread_rwlock_unlock(&rwlock);
            
            if (got == cache_map.end()) return -1;
            val = got->second;
            hits++;
            return 0;
        }
        void addCache(key_t key, value_t val) {
            if (cache_map.size()>200) return;
            //if (access_flag == false) return;

            if (pthread_rwlock_trywrlock(&rwlock)!=0) return;
            
            access_flag = false;

            cache_map[key] = val;
            pthread_rwlock_unlock(&rwlock);
            access_flag = true;

            
        }
*/
/*}}}*/
};


//-------------------------------------------------------------------------------




class HashCache2{
/*{{{*/
    //可以并行查 不能并行插
    public:
        std::unordered_map<key_t, value_t> cache_map;
        int hits;
        key_t cache_key = 0;

        int lookup(key_t key, value_t &val) {
            
            //std::cout<<"lookup!"<<key<<std::endl;
            auto got = cache_map.find(key);
            if (got==cache_map.end())
                return -1;
            val = got->second;
            hits++;
            return 0;
        }
        void addCache(key_t key, value_t val) {
            
            //if (cache_map.size()>100) return;
            if (cache_key == key) return;
            //std::cout<<"key "<<key<<"val "<<val<<std::endl;
            cache_map[key] = val;
            cache_key = key;

        }




/*}}}*/

};



#endif









