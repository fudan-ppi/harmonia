#ifndef MEMPOOL_H
#define MEMPOOL_H
#include<iostream>
#include<stdio.h>
#include<cassert>
#include<string.h>
const long KB = 1024;
const long MB = KB*1024;
const long GB = MB*1024;
const long init_mem_size = 8*GB;//4G

template<typename element_type>
class Mem_Pool{
    private:
        element_type *start;    //start address of Mem_Pool 
        element_type *end;      // next free address
        int size;               //Mem_Pool used element 
        long capability;         //Mem_Pool total element can be used
        long byte_capability;    //Mem_Pool pool total bytes;
    public:
        Mem_Pool(){
            start = (element_type*) malloc(init_mem_size);            
            if(start !=NULL){
                end = start;
                byte_capability = init_mem_size;
                capability = init_mem_size / sizeof(element_type);
                size = 0;
            }else{
                std::cout<<"Mempool malloc error"<<std::endl;
            }
        }

        ~Mem_Pool(){
            if(!start)
                free(start);
        }

        bool inPool(void *address){
           if((char *) start<= (char* )address && (char*)start+byte_capability >= (char*)address) 
               return true;
           else 
               return false;
        }

        element_type * getElement();
        element_type * elementAtIdx(int index){
            assert(index<size);
            return start+index;
        };
        int  indexAtAddress(element_type *p){
            assert(inPool(p));
            return  (p-start);
        };
        
        void getMetaForGPU(element_type *&start_address, unsigned int &total_byte_size){
            start_address = start;
            total_byte_size = size * sizeof(element_type);
        }
        int getSize() {
            return size;
        }
        element_type *getStart() {
            return start;
        }
        element_type *getEnd() {
            return end;
        }
        void moveBackOneSlot(int start);

        void reset(element_type *start1, element_type *end1){
           free(start);
           start = start1;
           end = end1;
           size = end-start;
        }

};

//------------------------------------------------------------------implementation
template<typename element_type>
element_type * Mem_Pool<element_type>::getElement(){
    element_type *newEle=  NULL ;
    if(size<capability){
        newEle = end;
        end++;
        size++;

    }else{
        //expansion ; reference vector implementation
        std::cout<<"Mem_Pool out of mem"<<std::endl;
    }
    return newEle;
}
template<typename element_type>
void Mem_Pool<element_type>::moveBackOneSlot(int start){
    element_type * src = elementAtIdx(start);
    element_type * dst = src+1;
    size_t s = (size-start -1)*sizeof(element_type);
    memmove(dst,src,s);
}

#endif
