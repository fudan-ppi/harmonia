#include <iostream>
#include <unordered_set>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include"regular_bpt.h"
#include"cfb.h"
#include "gpu-cfb.h"
#include "../src/conf.h"


using namespace std;
void search_cpu(ifstream &file, node *bpt_root){
	string s;
	key_t key;
	std::vector<key_t>  v;
	while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
		record * r = find(bpt_root,key,false);
		if(r==NULL){
			cout<<"Can not find"<<endl;
		}else{
			cout<<key<<" "<<r->value<<endl;
		}
	}
}
node* update(ifstream &file, node *bpt_root){
    std::vector<key_t> keys;
	string s;
	key_t key;
    struct timeval start,end;
	while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
        keys.push_back(key);
	}
    gettimeofday(&start,NULL);
    bpt_root = insert_parallel(bpt_root,keys); 

   // cout<<"Start Check"<<endl;
   // //test
   // for(auto k:keys){
   //     record * r = find(bpt_root,k,false);
   // 	if(r==NULL){
   // 		cout<<"Can not find"<<endl;
   // 	}else{
   // 		cout<<k<<" "<<r->value<<endl;
   // 	}
   // }


    gettimeofday(&end,NULL);
    double time =  (end.tv_sec-start.tv_sec) +(end.tv_usec-start.tv_usec)/1000000.0;
    cout<<"time:"<<time<<endl;
    return bpt_root;
}

node* update_batch(ifstream &file, node *bpt_root, int batch_size){
    std::vector<key_t> keys;
	string s;
	key_t key;
    struct timeval start,end;
	while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
        keys.push_back(key);
	}
    gettimeofday(&start,NULL);
    bpt_root = insert_parallel_batch(bpt_root,keys,batch_size); 

   // cout<<"Start Check"<<endl;
   // //test
   // for(auto k:keys){
   //     record * r = find(bpt_root,k,false);
   // 	if(r==NULL){
   // 		cout<<"Can not find"<<endl;
   // 	}else{
   // 		cout<<k<<" "<<r->value<<endl;
   // 	}
   // }


    gettimeofday(&end,NULL);
    double time =  (end.tv_sec-start.tv_sec) +(end.tv_usec-start.tv_usec)/1000000.0;
    cout<<"time:"<<time<<endl;
    return bpt_root;
}


void search_cpu_multithread(ifstream &file,node *bpt_root){
    struct timeval start,end;
    string s;
    key_t key;
    vector<key_t> keys;
    while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
        keys.push_back(key);
    }
    vector<val_t> vals(keys.size());
    cout<<keys.size()<<endl;
    gettimeofday(&start,NULL);
   
    omp_set_num_threads(OMP_Thread_num);
    //omp_set_num_threads(8);
    #pragma omp parallel for 
    for(int i=0;i<keys.size();i++){
        record * r = find(bpt_root,keys[i],false);
        if(r==NULL) continue;
        vals[i] = (r->value);
    }
    gettimeofday(&end,NULL);
        double time =  (end.tv_sec-start.tv_sec) +(end.tv_usec-start.tv_usec)/1000000.0;
    cout<<"time:"<<time<<endl;
}

void search_cpu_noleaf_multithread(ifstream &file,node *bpt_root){
    struct timeval start,end;
    string s;
    key_t key;
    vector<key_t> keys;
    while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
        keys.push_back(key);
    }
    vector<node *> vals(keys.size());
    cout<<keys.size()<<endl;
    gettimeofday(&start,NULL);
   
    omp_set_num_threads(OMP_Thread_num);
    //omp_set_num_threads(8);
    #pragma omp parallel for 
    for(int i=0;i<keys.size();i++){
        node * r = find_leaf(bpt_root,keys[i],false);
        if(r==NULL) continue;
        vals[i] = r;
    }
    gettimeofday(&end,NULL);
        double time =  (end.tv_sec-start.tv_sec) +(end.tv_usec-start.tv_usec)/1000000.0;
    cout<<"time:"<<time<<endl;
}

void search_prefix_tree_cpu(ifstream &file, CFB *bpt){
	string s;
	key_t key;
	std::vector<key_t> v;
	while(getline(file,s)){
		sscanf(s.c_str(), TYPE_D,&key);
		record * r = bpt->search(key,true);
	}

}
int main(int argc, char *argv[]){
	node * bpt_root=NULL;
	cout<<"Tree order is "<<order<<endl;
	// read insert file
	if(argc>1){
		ifstream insert_file;
		insert_file.open(argv[1]);
		assert(insert_file.is_open());
		string s;
		while(getline(insert_file,s)){
			key_t key;
			sscanf(s.c_str(), TYPE_D,&key);
			bpt_root = insert(bpt_root, key, key);
		}
		insert_file.close();
	}else{
		cout<<"insert file empty!"<<endl;
		exit(0);
	}
	//check search file
	if(argc>2){

	}else{
		cout<<"search file empty!"<<endl;
		exit(0);
	}

    
	//check search file
	if(argc>3){
		ifstream search_file;
		search_file.open(argv[2]);
		assert(search_file.is_open());
		switch(atoi(argv[3])){
			case 0:{
				search_cpu(search_file,bpt_root);
				break;
			}
            case 1:{
                CFB *cfb = generateCFB(bpt_root);
                multi_prefix_sort(*cfb, search_file);
                break;       
            }
            case 2:{
                CFB *cfb = generateCFB(bpt_root);
                search_prefix_tree_cpu(search_file,cfb);
                break;       
            }
            case 3:{
                search_cpu_multithread(search_file,bpt_root);
                break;
            } 
            case 4:{
                ifstream update_file;
                cout<<argv[4]<<endl;
                update_file.open(argv[4]);
                assert(update_file);
                bpt_root = update(update_file,bpt_root);
                break;
            }
            case 5:{
                CFB *cfb = generateCFB(bpt_root);
                multi_prefix_nosort(*cfb, search_file);
                break;
            }
            case 6:{
                CFB *cfb = generateCFB(bpt_root);
                cfb->generatePointerSection();
                multi_noprefix_nosort(*cfb, search_file);
                break;
            }
            case 7:{
                RB *rb = generateRB(bpt_root);
                multi_noprefix_nosort_RB(*rb, search_file);
                break;
            }
            case 8:{
                search_cpu_noleaf_multithread(search_file,bpt_root);
                break;
            }
            case 9:{
                CFB *cfb = generateCFB(bpt_root);
                half_multi_prefix_sort(*cfb, search_file);
                break;       
            }
            case 10:{
                CFB *cfb = generateCFB(bpt_root);
                half_multi_prefix_nosort(*cfb, search_file);
                break;
            }
            case 11:{
                RB *rb = generateRB(bpt_root);
                half_multi_noprefix_nosort_RB(*rb, search_file);
                break;
            }
            case 15:{
                ifstream update_file;
                cout<<argv[4]<<endl;
                update_file.open(argv[4]);
                assert(update_file);
                bpt_root = update_batch(update_file,bpt_root, atoi(argv[5]));
                break;
            }

            

		}
		search_file.close();
	}
}
