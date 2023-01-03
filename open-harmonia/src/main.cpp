#include<iostream>
#include<unordered_set>
#include<stdlib.h>
#include "bpt.h"
#include "mempool.h"
#include "gbpt.h"
#include "ppi-bpt.h"
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include "ppi-bpt.h"
#include "conf.h"
#include "cpu-search.h"
using namespace std;

void search_pipe(ifstream &file, BPlusTree &tree, const unordered_set<key_t> &S) ;
void search_cpu_test(ifstream &file, BPlusTree &tree) ;
void test_gbpt_update(BPlusTree &tree,ifstream &file);
void test_gbpt_update_batch(BPlusTree &tree,ifstream &file,int batch_size);
void test_gbpt_Newupdate(BPlusTree &tree,ifstream &file);
void test_gbpt_Newupdate_batch(BPlusTree &tree,ifstream &file, int, int);

#ifdef BOOST
void search_cpu_sort_test(ifstream &file, BPlusTree &tree);
#endif
int main(int argc, char *argv[]){
    
//    cout<<"void* size:          "<<sizeof(void *)<<endl;
//    cout<<"int size:            "<<sizeof(int)<<endl;
//    cout<<"long size:           "<<sizeof(long)<<endl;
//    cout<<"key_t size:      "<<sizeof(key_t)<<endl;
//    cout<<"Inner_node size:     "<<sizeof(Inner_node)<<endl;
//    cout<<"Inner_metadata size: "<<sizeof(Inner_meta)<<endl;
//    cout<<"Record size:         "<<sizeof(Record)<<endl;
//    cout<<"Record size:         "<<sizeof(Record_list)<<endl;
//    cout<<"Leaf_node size:      "<<sizeof(BLeaf_node)<<endl;
//    cout<<"Mem_Pool size:       "<<sizeof(Mem_Pool<Inner_node>)<<endl;
//    cout<<"BPlusTree size:      "<<sizeof(BPlusTree)<<endl;
     
    BPlusTree tree;
    
    unordered_set<key_t> S;
    int insert_len =    1<<23;
    int search_size =   100000000;
    //struct timeval start;
    //struct timeval end;
    //double tt=0;


    if (argc > 1) {
        ifstream insert_file;
        insert_file.open(argv[1]);
        assert(insert_file.is_open());
        string s;
        insert_len = 0;
        while(getline(insert_file,s)) {
            key_t key;
            sscanf(s.c_str(), TYPE_D, &key);
            //cout<<key<<endl;
            //if(key == 1987631206){
            //    cout<<"ha"<<endl;
            //}
            tree.insert(key,key);
            S.insert(key);
            insert_len++;
        }
        insert_file.close();

    }else {
        cout<<"insert file empty!"<<endl;
        exit(0);
//        for(int i=0;i<insert_len;i++){
//            key_t key = (rand());
//            if(!tree.insert(key,key)){
//                    continue;
//                if(S.find(key)!=S.end()){
//                    //cout<<"wrong"<<endl;
//                }
//            }
//            S.insert(key);
//        }
    }

    

    if (argc>2) {
        ifstream search_file;
        search_file.open(argv[2]);
        assert(search_file.is_open());
        string s;
        search_size = 0;

        
        //cpu search
        /*
        int i = 0;
        vector<key_t*> v;
        key_t *pkey = (key_t *)malloc(P*sizeof(key_t));
        while(getline(search_file,s)) {
            key_t key;
            sscanf(s.c_str(), TYPE_D, &key);
            gettimeofday(&start, NULL);
            value_t a = tree.search(key);
            gettimeofday(&end, NULL);
            tt += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;

            search_size++;
           //check  
           //auto b = S.find(key);
           //if (b!=S.end()){  
           //    if(a!=*b) exit(0);
           //    cout<<a<<endl;
           //    cout<<*b<<endl;
           //}
           //else {
           //    assert(a==-1);
           //}
            
        }
        */
    }
    else {
        cout<<"search file empty!"<<endl;
        exit(0);

//        for(int j=0;j<1;j++){
//            for (int i=0;i<search_size ; i++) {
//                key_t k1 = rand();
//                //key_t k2 = k1 + rand()%1000000;
//                gettimeofday(&start, NULL);
//                value_t a = tree.search(k1);
//                gettimeofday(&end, NULL);
//                tt += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
//                /*
//                auto b = S.find(k1);
//                if (b!=S.end()){  
//                    if(a!=*b) exit(0);
//                    cout<<a<<endl;
//                    cout<<*b<<endl;
//                }else {
//    //                cout<<a<<endl;
//                    assert(a==-1);
//                }
//    */
//            }
//        }
    }


    //tree.traverse(tree.getRoot());
    BPlusTree *new_tree;
    tree.sort(new_tree);
//    cout<<"Tree Height: "<<new_tree->getHeight()<<endl;
//    cout<<"Tree innernode_size"<<new_tree->getInnerSize()<<endl;
//    new_tree->print_LevelNum();
//    new_tree->empty_cal();
//


    if (argc > 3) {
   
        ifstream search_swp_file26;
        search_swp_file26.open(argv[2]);
        assert(search_swp_file26.is_open());
        
        switch (atoi(argv[3])) {
            case 0:
                test_prefix_8thread(*new_tree,search_swp_file26);
                break;
            case 1:
                test_prefix_2thread(*new_tree,search_swp_file26);
                break;
            case 2:
                test_prefix_sort_8thread(*new_tree,search_swp_file26);
                break;
            case 3:
                test_sort_8thread(tree, search_swp_file26);
                break;
            case 4:
                PPI_BPT_Search_GPU_V9_thread_scheduling(*new_tree, search_swp_file26);  //test_prefix_sort_2thread
                break;
            case 5:
                test_8thread(tree, search_swp_file26);
                break;
            case 6:
                test_2thread(tree, search_swp_file26);
                break;
            case 7:
                test_sort_2thread(tree, search_swp_file26);
                break;

            case 8: //test
                PPI_BPT_Search_GPU_V14_1_prefix_up_3_level_constant_down_readonly(*new_tree, search_swp_file26);
                break;
            case 9:
                PPI_BPT_Search_GPU_V14_2_prefix_up_full_constant_down_readonly(*new_tree, search_swp_file26);
                break;

            case 10:
                PPI_BPT_Search_GPU_V10_multigpu_basedv5(*new_tree, search_swp_file26);  //multi-gpu ppi 
                break;
            case 11:
                BPT_Search_GPU_multi_gpu_v4(tree,search_swp_file26);                    //multi-gpu hb+
                break;
            case 12:{
                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_update(tree,update_file);
                    break;
                }
            case 13:{
                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_update(*new_tree,update_file);
                    break;
                }

            case 14:
                PPI_BPT_Search_GPU_V14_3_prefix_opt(*new_tree, search_swp_file26);
                break;
            case 15:
                compare_times_2thread(*new_tree, search_swp_file26);
                break;




            case 16:
                multi_prefix_sort_2thread(*new_tree, search_swp_file26);
                break;
            case 17:
                multi_prefix_nosort_2thread(*new_tree, search_swp_file26);
                break;
            case 18:
                multi_prefix_nosort_8thread(*new_tree, search_swp_file26);
                break;
            case 19:
                multi_noprefix_nosort_8thread(tree, search_swp_file26);
                break;
            case 20:
                multi_prefix_sort_4thread(*new_tree, search_swp_file26);
                break;
            case 21:
                multi_prefix_sort_1thread(*new_tree, search_swp_file26);
                break;
            case 22:
                multi_prefix_sort_8thread(*new_tree, search_swp_file26);
                break;
            case 23:{
                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_Newupdate(*new_tree,update_file);
                    break;
                }

            case 24:
                compare_times_1thread(*new_tree, search_swp_file26);
                break;
            case 25:
                compare_times_4thread(*new_tree, search_swp_file26);
                break;
            case 26:
                multi_prefix_nosort_1thread(*new_tree, search_swp_file26);
                break;
            case 27:
                multi_prefix_nosort_4thread(*new_tree, search_swp_file26);
                break;
            case 28:
                full_multi_prefix_sort_1thread(*new_tree, search_swp_file26);
                break;
            case 29:
                full_multi_prefix_nosort_1thread(*new_tree, search_swp_file26);
                break;
            case 30:
                full_multi_prefix_sort_8thread(*new_tree, search_swp_file26);
                break;
            case 31:
                full_multi_prefix_nosort_8thread(*new_tree, search_swp_file26);
                break;
            case 33:
                bit_multi_prefix_sort_8thread(*new_tree, search_swp_file26,atoi(argv[4]),atoi(argv[5]));
                break;
            case 34:
                bit_multi_prefix_sort_1thread(*new_tree, search_swp_file26,atoi(argv[4]),atoi(argv[5]));
                break;
            

            case 35:    //old serial version: up sort down, in ppi-bpt.cu.  v1: return result in same sequence with input key
                PPI_BPT_Search_GPU_basic(tree, search_swp_file26);
                break;
            case 36:    //old serial version: up sort down, in ppi-bpt.cu   v2: return key and result in a new sequence
                PPI_BPT_Search_GPU_basic_v2(tree, search_swp_file26);
                break;
            case 37:    //old serial version: up sort down, in ppi-bpt.cu   v2: return key and result in a new sequence
                PPI_BPT_Search_GPU_V2_8thread(tree, search_swp_file26, atoi(argv[4]), atoi(argv[5]));
                break;

            
            case 38:{
                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_Newupdate(*new_tree, update_file);
                    multi_prefix_sort_1thread_noleaf(*new_tree, search_swp_file26);
                    //multi_prefix_sort_1thread(*new_tree, search_swp_file26);
                    break;
                }
            case 39:{
                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_Newupdate_batch(*new_tree, update_file, atoi(argv[5]), atoi(argv[6]));
                 //   multi_prefix_sort_1thread_noleaf(*new_tree, search_swp_file26);
                    //multi_prefix_sort_1thread(*new_tree, search_swp_file26);
                    break;

                    }
            case 40:{

                    ifstream update_file;
                    update_file.open(argv[4]);
                    assert(update_file.is_open());
                    test_gbpt_update_batch(tree, update_file, atoi(argv[5]));
                    break;
                    }
            case 41: {
                        search_cpu_test(search_swp_file26,tree);
                        break;
                     }
            case 42: {
                        search_cpu_test( search_swp_file26, *new_tree);
                        break;
                     }
            case 43:
                    full_multi_prefix_sort_1thread_notransfer(*new_tree, search_swp_file26);
                    break;
            case 44:
                    full_multi_prefix_sort_8thread_notransfer(*new_tree, search_swp_file26);
                    break;
            case 45:
                    full_multi_prefix_nosort_8thread_notransfer(*new_tree, search_swp_file26);
                    break;
            case 46:
                    full_multi_prefix_nosort_1thread_notransfer(*new_tree, search_swp_file26);
                    break;
            case 47:
                    HB_simple(tree, search_swp_file26);
                    break;
            case 48:
                    HB_simple_notransfer(tree, search_swp_file26);
                    break;
            case 49:
                    range_prefix_sort_1thread_search_and_1thread_scan(*new_tree, search_swp_file26);
                    break;
            case 50:
                    range_prefix_sort_1thread_search_and_1thread_scan_notransfer(*new_tree, search_swp_file26);
                    break;
#ifdef BOOST
            case 51:
                    search_cpu_sort_test(search_swp_file26, *new_tree);
                    break;
            case 52:
                    search_cpu_sort_test(search_swp_file26, tree);
                    break;
            case 53:
                    search_cpu_prefetch_sort(search_swp_file26, *new_tree);
                    break;
            case 54:
                    search_cpu_prefetch_sort(search_swp_file26, tree);
                    break;
            case 55:
                    search_cpu_prefetch_sort_NTG(search_swp_file26, *new_tree);
                    break;
            case 56:
                    search_cpu_prefetch_sort_NTG(search_swp_file26, tree);
                    break;
#endif
            case 57:
                    search_cpu_prefetch(search_swp_file26, *new_tree);
                    break;
            case 58:
                    search_cpu_prefetch(search_swp_file26, tree);
                    break;
            
#ifdef BOOST
            case 59:
                    search_range_cpu_prefetch_sort_NTG(search_swp_file26, *new_tree);
                    break;
#endif



            case 60:
                    range_hb(tree, search_swp_file26); //range query for hb 
                    break;
            case 61:
                    range_hb_notransfer(tree, search_swp_file26); //range query for hb 
                    break;
            case 62:
                    search_range_cpu_prefetch_HB(search_swp_file26, tree);
                    break;

#ifdef BOOST
            case 63:
                    search_cpu_prefetch_NTG(search_swp_file26, *new_tree);
                    break;
            case 64:
                    search_cpu_prefetch_NTG(search_swp_file26, tree);
                    break;
#endif
        }   

        exit(0);
    }






/*{{{*/
    // cout<<"====================================="<<endl;
   // cout<<"HB+ tree serial"<<endl;
   // cout<<"CPU time:    "<<tt<<endl;
   // cout<<"Tree Height: "<<tree.getHeight()<<endl;
   // cout<<"insert_len:  "<<insert_len<<endl;
   // cout<<"search_szie: "<<search_size<<endl;


   // cout<<"====================================="<<endl;
   // //search software pipe line
    ifstream search_swp_file;
    search_swp_file.open(argv[2]);
    assert(search_swp_file.is_open());
    search_pipe(search_swp_file,tree,S);
    search_swp_file.close();
//
//
   // cout<<"====================================="<<endl;
   // //GPU 
   // ifstream gpu_search_file;
   // gpu_search_file.open(argv[2]);
   // assert(gpu_search_file.is_open());
   // BPT_Search_GPU(tree,gpu_search_file);
   // gpu_search_file.close();
 
   // cout<<"====================================="<<endl;
   // ifstream search_swp_file1;
   // search_swp_file1.open(argv[2]);
   // assert(search_swp_file1.is_open());
   // BPT_Search_GPU_DoubleBuffering(tree,search_swp_file1);
   //// 
 
  //  cout<<"====================================="<<endl;
  //  ifstream search_swp_file2;
  //  search_swp_file2.open(argv[2]);
  //  assert(search_swp_file2.is_open());
  //  BPT_Search_GPU_DoubleBuffering_v2(tree,search_swp_file2);
  //  
  //   cout<<"====================================="<<endl;
  //   cout<<"HB+ tree [double buffer load balance]"<<endl;
  //   ifstream search_swp_file3;
  //   search_swp_file3.open(argv[2]);
  //   assert(search_swp_file3.is_open());
  //   BPT_Search_GPU_DoubleBuffering_v3(tree,search_swp_file3);
 //
/* 
    cout<<"====================================="<<endl;
    cout<<"gpu_ppi_basic_serial"<<endl;
    ifstream search_swp_file4;
    search_swp_file4.open(argv[2]);
    assert(search_swp_file4.is_open());
    PPI_BPT_Search_GPU_basic(tree, search_swp_file4);


    cout<<"====================================="<<endl;
    cout<<"gpu_ppi_basic_serial_v2 [key order change]"<<endl;
    ifstream search_swp_file5;
    search_swp_file5.open(argv[2]);
    assert(search_swp_file5.is_open());
    PPI_BPT_Search_GPU_basic_v2(tree, search_swp_file5);
*/

    
    //cout<<"====================================="<<endl;
    //ifstream search_swp_file6;
    //search_swp_file6.open(argv[2]);
    //assert(search_swp_file6.is_open());
    //PPI_BPT_Search_GPU_V1_batch(tree, search_swp_file6);


//  cout<<"====================================="<<endl;
//  ifstream search_swp_file7;
//  search_swp_file7.open(argv[2]);
//  assert(search_swp_file7.is_open());
//  PPI_BPT_Search_GPU_V2_8thread(tree, search_swp_file7);

    //cout<<"====================================="<<endl;
    //ifstream search_swp_file8;
    //search_swp_file8.open(argv[2]);
    //assert(search_swp_file8.is_open());
    //PPI_BPT_Search_GPU_V2_4thread(tree, search_swp_file8);
   

    //cout<<"====================================="<<endl;
    //ifstream search_swp_file9;
    //search_swp_file9.open(argv[2]);
    //assert(search_swp_file9.is_open());
    //PPI_BPT_Search_GPU_V3_2thread(tree, search_swp_file9);
  
    
   // cout<<"====================================="<<endl;
   // ifstream search_swp_file10;
   // search_swp_file10.open(argv[2]);
   // assert(search_swp_file10.is_open());
   // PPI_BPT_Search_GPU_V4_2thread(tree, search_swp_file10);
    
   // v4 版本一次只能跑一个 
//   cout<<"====================================="<<endl;
//   ifstream search_swp_file11;
//   search_swp_file11.open(argv[2]);
//   assert(search_swp_file11.is_open());
//   PPI_BPT_Search_GPU_V4_2thread_whole(*new_tree, search_swp_file11);
 
      

    /*
    cout<<"====================================="<<endl;

    ifstream search_swp_file12;
    search_swp_file12.open(argv[2]);
    assert(search_swp_file12.is_open());
    PPI_BPT_Search_GPU_V4_2thread_whole_cpusort(tree, search_swp_file12);
    */
    
 //  cout<<"====================================="<<endl;
 //  ifstream search_swp_file13;
 //  search_swp_file13.open(argv[2]);
 //  assert(search_swp_file13.is_open());
 //  PPI_BPT_Search_GPU_V5_2thread_new_tree(*new_tree, search_swp_file13);
//
//   cout<<"====================================="<<endl;
//   ifstream search_swp_file14;
//   search_swp_file14.open(argv[2]);
//   assert(search_swp_file14.is_open());
//   PPI_BPT_Search_GPU_V5_2thread_new_tree_serial(*new_tree, search_swp_file14);
 
 //  cout<<"====================================="<<endl;
  // ifstream search_swp_file15;
  // search_swp_file15.open(argv[2]);
  // assert(search_swp_file15.is_open());
  // PPI_BPT_Search_GPU_V5_2thread_new_tree_CPUMultiThread(*new_tree, search_swp_file15);

  // cout<<"====================================="<<endl;
  // ifstream search_swp_file16;
  // search_swp_file16.open(argv[2]);
  // assert(search_swp_file16.is_open());
  // PPI_BPT_Search_GPU_V6_balance(*new_tree, search_swp_file16);
    
//   cout<<"====================================="<<endl;
  //ifstream search_swp_file17;
  //search_swp_file17.open(argv[2]);
  //assert(search_swp_file17.is_open());
  //PPI_BPT_Search_GPU_V7_2thread_new_tree_CPUMultiThread(*new_tree, search_swp_file17);

  //   cout<<"====================================="<<endl;
 //  ifstream search_swp_file18;
 //  search_swp_file18.open(argv[2]);
 //  assert(search_swp_file18.is_open());
 //  PPI_BPT_Search_GPU_V8_thread_scheduling(*new_tree, search_swp_file18);

  //  ifstream search_swp_file19;
  //  search_swp_file19.open(argv[2]);
  //  assert(search_swp_file19.is_open());                                   
  //  PPI_BPT_Search_GPU_V9_thread_scheduling(*new_tree, search_swp_file19);
  
//测试PPI-tree multi GPU
//     ifstream search_swp_file20;
//     search_swp_file20.open(argv[2]);
//     assert(search_swp_file20.is_open());                                   
//     PPI_BPT_Search_GPU_V10_multigpu_basedv5(*new_tree, search_swp_file20);
  
//测试HB+ tree mult GPU
  //     cout<<"====================================="<<endl;
  //     ifstream search_swp_file21;
  //     search_swp_file21.open(argv[2]);
  //     assert(search_swp_file21.is_open());
  //     BPT_Search_GPU_multi_gpu_v4(tree,search_swp_file21);
 // 
 
 //  cout<<"====================================="<<endl;
 //  ifstream search_swp_file22;
 //  search_swp_file22.open(argv[2]);
 //  assert(search_swp_file22.is_open());
 //  PPI_BPT_Search_GPU_V11_multigpu_basedv5_cache(*new_tree,search_swp_file22);

  // ifstream search_swp_file23;
  // search_swp_file23.open(argv[2]);
  // assert(search_swp_file23.is_open());
  // PPI_BPT_Search_GPU_V12_thread_scheduling_cpu_cache(*new_tree,search_swp_file23);

    //ifstream search_swp_file24;
    //search_swp_file24.open(argv[2]);
    //assert(search_swp_file24.is_open());
    //PPI_BPT_Search_GPU_V13_thread_scheduling_cpu_cache_profile_version(*new_tree,search_swp_file24);

//    ifstream search_swp_file25;
//    search_swp_file25.open(argv[2]);
//    assert(search_swp_file25.is_open());
//    test_prefix_2thread(*new_tree,search_swp_file25);
//


   //  ifstream search_swp_file26;
   //  search_swp_file26.open(argv[2]);
   //  assert(search_swp_file26.is_open());
   //  test_prefix_8thread(*new_tree,search_swp_file26);
   //   /*}}}*/
}
// batch size is file size 
void test_gbpt_update(BPlusTree &tree,ifstream &file){
   /*{{{*/
    vector<key_t> keys;
    keys.reserve(512*1000);
    string s;
    key_t key;
    while(getline(file,s)){
        sscanf(s.c_str(),TYPE_D,&key);
        keys.push_back(key);
    }
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    tree.insert_parallel(keys,keys);
    gettimeofday(&end, NULL);
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    cout<<"update time:\t"<<tt<<endl;
    /*}}}*/
}
void test_gbpt_update_batch(BPlusTree &tree,ifstream &file,int batch_size){
/*{{{*/
    vector<key_t> keys;
    keys.reserve(512*1000);
    string s;
    key_t key;
    while(getline(file,s)){
        sscanf(s.c_str(),TYPE_D,&key);
        keys.push_back(key);
    }
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    tree.insert_parallel_batch(keys,keys,batch_size);
    gettimeofday(&end, NULL);
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    cout<<"update time:\t"<<tt<<endl;
    /*}}}*/
}

// batch size is file size 
void test_gbpt_Newupdate(BPlusTree &tree,ifstream &file){
/*{{{*/
    vector<key_t> keys;
    keys.reserve(512*1000);
    string s;
    key_t key;
    while(getline(file,s)){
        sscanf(s.c_str(),TYPE_D,&key);
        keys.push_back(key);
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    tree.insert_parallel_Newupdate(keys,keys);
    gettimeofday(&end, NULL);
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    cout<<"update time:\t"<<tt<<endl;
    /*}}}*/
}
void test_gbpt_Newupdate_batch(BPlusTree &tree,ifstream &file, int syn_batch_size, int rebuild_batch_size){
/*{{{*/
    vector<key_t> keys;
    keys.reserve(512*1000);
    string s;
    key_t key;
    while(getline(file,s)){
        sscanf(s.c_str(),TYPE_D,&key);
        keys.push_back(key);
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    tree.insert_parallel_Newupdate_batch(keys,keys,syn_batch_size, rebuild_batch_size);
    gettimeofday(&end, NULL);
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    cout<<"update time:\t"<<tt<<endl;
    /*}}}*/
}


//search_pipe is not available for Harmonia!!!!!!!! 
void search_pipe(ifstream &file, BPlusTree &tree, const unordered_set<key_t> &S) {
    //vector<value_t> values(P);/*{{{*/
    
    
    struct timeval start;
    struct timeval end;
    double tt=0;

    string s;
    key_t key;
    vector<vector<key_t>> keys;
    vector<key_t> tmp;
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        tmp.push_back(key);
        if(tmp.size()==P){
            keys.push_back(tmp);
            tmp.clear();
        }
    }
    int vkeys_size = keys.size();




       
    gettimeofday(&start, NULL);
    //    omp_set_num_threads(32);
    #pragma omp parallel for
    for (int i=0;i<vkeys_size;i++) {
        

        vector<value_t> values(P);
        tree.search_swp(keys[i], values);
       //search_swp is not available for Harmonia!!!!!!!! 

#ifdef ENABLE_TEST 
        for (int j=0;j<P;j++) {
            cout<<keys[i][j]<<": "<<values[j]<<endl;
        }
        //auto v = keys[i];
        //for (int j=0;j<P;j++) {
        //    if (S.find(v[j])!=S.end()) {
        //        cout<<v[j]<<": "<<values[j]<<endl;
        //        assert(values[j]==v[j]);
        //    }else
        //        assert(values[j]==-1);
        //}
        //cout<<i<<"  over"<<endl;
#endif
    }
   

    gettimeofday(&end, NULL);

    
    tt += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    cout<<"HB+ SWP [no gpu]:"<<endl;
    cout<<"Search_swp_size: "<<vkeys_size*P<<endl;
    cout<<"Search_swp time:      "<<tt<<endl;/*}}}*/
}


//for HB and Harmonia
void search_cpu_test(ifstream &file, BPlusTree &tree) {
/*{{{*/    
    
    struct timeval start;
    struct timeval end;
    double tt=0;

    string s;
    key_t key;
    vector<key_t> keys;
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        keys.push_back(key);
    }
    int keys_size = keys.size();

       
    vector<value_t> values(keys_size);
   
    gettimeofday(&start, NULL);
    //    omp_set_num_threads(32);
    #pragma omp parallel for
    for (int i=0;i<keys_size;i++) {
        values[i] = tree.search(keys[i]);
    }

    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    
    tt += (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu]: "<<endl;
    else cout<<"HB+ [no gpu]:"<<endl;
    cout<<"Search_size: "<<keys_size<<endl;
    cout<<"Search time:      "<<tt<<endl;/*}}}*/
}


#ifdef BOOST
#include <boost/sort/spreadsort/spreadsort.hpp>
#define BUCKET_SIZE 100000
using namespace boost::sort::spreadsort;
void search_cpu_sort_test(ifstream &file, BPlusTree &tree) {
    /*{{{*/
    struct timeval start;
    struct timeval end;

    string s;
    key_t key;
    vector<key_t> keys;
    
    //build 
    while(getline(file,s)){
        sscanf(s.c_str(), TYPE_D, &key);
        keys.push_back(key);
    }

    long keys_size = keys.size();
    int total = keys_size / BUCKET_SIZE; 
 
    gettimeofday(&start, NULL);
    #pragma omp parallel for
    for (int i=0;i<total;i++) {
        integer_sort(&(keys[i*BUCKET_SIZE]), &(keys[(i+1)*BUCKET_SIZE]), rightshift());
    }
    gettimeofday(&end, NULL);
    double sort_time =  (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;

    vector<value_t> values(keys_size);

   
    gettimeofday(&start, NULL);


    #pragma omp parallel for 
    for (int i=0;i<total;i++) {
        for (int j=0;j<BUCKET_SIZE;j++) {
            values[i*BUCKET_SIZE+j] = tree.search(keys[i*BUCKET_SIZE+j]);
        }
    }

    gettimeofday(&end, NULL);

#ifdef ENABLE_TEST
    for (int i=0;i<keys_size;i++) {
        cout<<"key: "<<keys[i]<<" end: "<<values[i]<<endl;
    }
#endif
    double tt = (end.tv_sec - start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000.0 ;
    if (tree.isPPItree) cout <<"Harmonia [no gpu] cpu-sort "<<endl;
    else cout<<"HB+ [no gpu]  cpu-sort"<<endl;
    cout<<"Search_size "<<keys_size<<endl;
    cout<<"Sort time:      "<<sort_time<<endl;
    cout<<"Search time:      "<<tt<<endl;
    cout<<"Total time:      "<<tt+sort_time<<endl;
    /*}}}*/
}
#endif
