#include<iostream>
#include<vector>
#include<pthread.h>
#include<tuple>
#include<algorithm>
class Chunk{
    public:
        volatile bool flag;
        std::tuple<int,int,int> data;
    Chunk(){
        flag = false;
    }

    Chunk(int start,int end,int bucket_idx){
        flag =  true;
        data = std::make_tuple(start,end,bucket_idx);   
    }
};
class Thread_To_LBBuffer{
    public:
    pthread_mutex_t lock;
    int mapping ;
    int process_idx;
    Thread_To_LBBuffer(){
        mapping = 0;
        process_idx = 0;
        pthread_mutex_init(&lock,NULL);
    }
    std::pair<int,int> getMappingAndProcessIdx(){
        pthread_mutex_lock(&lock);
        std::pair<int,int> ret;
        ret.first = mapping;
        ret.second = process_idx;
        pthread_mutex_unlock(&lock);
        return ret;
    }
    void setProcessIdx(int _mapping, int idx){
        pthread_mutex_lock(&lock);
        if(mapping == _mapping) process_idx = idx;
        pthread_mutex_unlock(&lock);
    }

    void setMappingAndProcessIdx(int x,int idx){
        pthread_mutex_lock(&lock);
        mapping=x;
        process_idx = idx;
        pthread_mutex_unlock(&lock);
    }
    ~Thread_To_LBBuffer(){
        pthread_mutex_destroy(&lock);
    }
};

class LBBuffer{//Low_Blocking_buffer
    public:
        std::vector<Chunk> buffer_queue;
        std::vector<Thread_To_LBBuffer *> thread_array;
        volatile int tail;
        
        LBBuffer(){
            tail = 0;
            buffer_queue.reserve(100000);
        }
        bool put(int start,int end,int bucket_idx){
            buffer_queue.push_back(Chunk(start,end,bucket_idx));
            tail++;
            return true;
        }
        std::tuple<int,int,int> get_unblock(int &idx){
            while(1){
                if(idx >= tail) {
                    //std::cout<<"haha"<<std::endl;
                    std::tuple<int, int, int> tmp_data = std::make_tuple(-2,0,0);  
                    return tmp_data;
                }

                if(idx == tail-1){
                    if(std::get<0>(buffer_queue[idx].data) == -1)return buffer_queue[idx].data;
                }
                if(__sync_bool_compare_and_swap(&(buffer_queue[idx].flag),true,false)){
                    idx++;
                    return buffer_queue[idx-1].data;
                }

                idx++;
            }
        } 
        std::tuple<int,int,int> get(int &idx){
            while(1){
                if(idx >= tail) {
                    //std::cout<<"haha"<<std::endl;
                    
                    continue; 
                }

                if(idx == tail-1){
                    if(std::get<0>(buffer_queue[idx].data) == -1)return buffer_queue[idx].data;
                }
                if(__sync_bool_compare_and_swap(&(buffer_queue[idx].flag),true,false)){
                    idx++;
                    return buffer_queue[idx-1].data;
                }

                idx++;
            }
        }

        void setFinish(){
            buffer_queue.push_back(Chunk(-1,-1,-1));
            tail++;
        }

        int getThreadNum() {return thread_array.size();}

        void pushThread(Thread_To_LBBuffer * tp) {
            thread_array.push_back(tp);
        }

        Thread_To_LBBuffer *popThread() {
            Thread_To_LBBuffer *tp = thread_array.back();
            thread_array.pop_back();
            return tp;
        }
        int getNextProcessIdx(){
            int m = 0;
            for(auto tp : thread_array){
                m = std::max(tp->process_idx,m);
            }
            return m;
        }
        int getWorkLeft() {
            int m = getNextProcessIdx();
            //std::cout<<"tail: "<<tail<<" head: "<<m<<std::endl;
            return tail - m;
        }

};













//======================================================================================
class Buffer{ //buffer use global lock
    public:
    std::vector<std::tuple<int,int,int>> buffer_queue;
    int head;
    int tail;
    std::vector<int> thread_array;
    pthread_mutex_t head_lock;
    pthread_cond_t qready;
    Buffer(){
        head = 0;
        tail = 0;
        pthread_mutex_init(&head_lock,NULL);
        pthread_cond_init(&qready,NULL);
    };
    void push(int start,int end,int bucket_idx){//just one producter
        buffer_queue.push_back(std::make_tuple(start,end,bucket_idx));
        tail++;
        pthread_cond_broadcast(&qready);
    }

    std::tuple<int,int,int> pop(){//multi-comsumer
        //判断是否有可读的
        pthread_mutex_lock(&head_lock);
        while(head==tail )
            pthread_cond_wait(&qready,&head_lock);

        std::tuple<int,int,int> ret = buffer_queue[head];
        if(std::get<0>(ret)!=-1)
            head++;
        
        pthread_mutex_unlock(&head_lock);
        return ret;
    }
    void setFinish(){
        buffer_queue.push_back(std::make_tuple(-1,-1,-1));
        tail++;
        pthread_cond_broadcast(&qready);
    }
    
    int getThreadNum() {return thread_array.size();}
    void pushThread(int threadIdx) {
        thread_array.push_back(threadIdx);
    }
    int popThread() {
        int temp = thread_array.back();
        thread_array.pop_back();
        return temp;
    }
    int getWorkLeft() {

        //std::cout<<"tail: "<<tail<<" head: "<<head<<std::endl;
        return tail - head;
    }



    ~Buffer(){
        pthread_cond_destroy(&qready);
        pthread_mutex_destroy(&head_lock);
    };
};

class ThreadToBuffer{
    public:
    pthread_mutex_t lock;
    int mapping ;
    ThreadToBuffer(){
        mapping = 0;
        pthread_mutex_init(&lock,NULL);
    }
    int getMapping(){
        int ret;
        pthread_mutex_lock(&lock);
        ret = mapping;
        pthread_mutex_unlock(&lock);
        return ret;
    }
    void setMapping(int x){
        pthread_mutex_lock(&lock);
        mapping=x;
        pthread_mutex_unlock(&lock);
    }
    ~ThreadToBuffer(){
        pthread_mutex_destroy(&lock);
    }
};
