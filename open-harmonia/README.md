# Open Harmonia

By running `sh harmonia/startup.sh` on your host machine, a runnable environment for Harmonia will be started. This README is used to show how to compile and run Harmonia inside a container.

## Compile

* To re-compile Harmonia, you can just run `sh build.sh` under the directory `/harmonia/open-harmonia`.
* If you want to make changes to the compile workflow, you should update the `CMakeLists.txt` and `build.sh`.

## Run

* Basically, the command to run Harmonia is organized as `./bpt_test {insert_file_path} {search_file_path} {option} [update_file]`. Please make sure, you are now under the directory `/harmonia/open-harmonia`.
* All pre-generated test data files are located in `/harmonia/open-harmonia/script/generate-data/dataset`. You can generate your own test data files by using `/harmonia/open-harmonia/script/generate-data/generate.py`
* `test.sh & tests/*.sh`
  * To run tests for Harmonia, you can run `sh /harmonia/open-harmonia/test.sh`. We use `/harmonia/open-harmonia/script/generate-data/dataset/insert64_4m/insert64_4m.txt` as default insert data file, `/harmonia/open-harmonia/script/generate-data/dataset/search64_1m/search64_uniform_1m.txt` as default search data file.
  * Some tests may cause core dump exception, that may because your `search_file` is not large enough. Try at least 100,000,000 lines `search_file` may help.(see `test.sh` and `tests/*.sh`)
  * Why `harmonia/open-harmonia/script/tests/{xxx}.sh` can not run?
  	* Because the original generated data files are located in `harmonia/open-harmonia/script/generate-data/dataset`, but in these shell test scripts, all data files should be located in `harmonia/open-harmonia/data_set`
  	* These tests are just examples. If you want to try to run them, you should move the `harmonia/open-harmonia/script/generate-data/dataset` to `harmonia/open-harmonia/data_set`, and then check those file paths in these shell scripts.

| option | test case                                                    | example                                                      | notes                                                        |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0      | test_prefix_8thread                                          | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 1      | test_prefix_2thread                                          | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 5      | test_8thread                                                 | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 6      | test_2thread                                                 | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 11     | BPT_Search_GPU_multi_gpu_v4                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 16     | multi_prefix_sort_2thread                                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 17     | multi_prefix_nosort_2thread                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 18     | multi_prefix_nosort_8thread                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 19     | multi_noprefix_nosort_8thread                                | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 20     | multi_prefix_sort_4thread                                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 21     | multi_prefix_sort_1thread                                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 22     | multi_prefix_sort_8thread                                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 24     | compare_times_1thread                                        | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 25     | compare_times_4thread                                        | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 26     | multi_prefix_nosort_1thread                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 27     | multi_prefix_nosort_4thread                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 28     | full_multi_prefix_sort_1thread                               | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 29     | full_multi_prefix_nosort_1thread                             | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 30     | full_multi_prefix_sort_8thread                               | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 31     | full_multi_prefix_nosort_8thread                             | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 35     | PPI_BPT_Search_GPU_basic                                     | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 36     | PPI_BPT_Search_GPU_basic_v2                                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 41     | search_cpu_test                                              | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 42     | search_cpu_test                                              | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 43     | full_multi_prefix_sort_1thread_notransfer                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 44     | full_multi_prefix_sort_8thread_notransfer                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 45     | full_multi_prefix_nosort_8thread_notransfer                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 46     | full_multi_prefix_nosort_1thread_notransfer                  | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 47     | HB_simple                                                    | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 48     | HB_simple_notransfer                                         | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 49     | range_prefix_sort_1thread_search_and_1thread_scan            | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 50     | range_prefix_sort_1thread_search_and_1thread_scan_notransfer | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 51     | search_cpu_sort_test                                         | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 52     | search_cpu_sort_test                                         | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 53     | search_cpu_prefetch_sort                                     | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 54     | search_cpu_prefetch_sort                                     | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 55     | search_cpu_prefetch_sort_NTG                                 | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 56     | search_cpu_prefetch_sort_NTG                                 | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 57     | search_cpu_prefetch                                          | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 58     | search_cpu_prefetch                                          | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 59     | search_range_cpu_prefetch_sort_NTG                           | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 60     | range_hb                                                     | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 61     | range_hb_notransfer                                          | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 62     | search_range_cpu_prefetch_HB                                 | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 63     | search_cpu_prefetch_NTG                                      | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 64     | search_cpu_prefetch_NTG                                      | `./bpt_test {insert_file} {search_file} {option}`            |                                                              |
| 12     | test_gbpt_update                                             | `./bpt_test {insert_file} {search_file} {option} {update_file}` |                                                              |
| 13     | test_gbpt_update                                             | `./bpt_test {insert_file} {search_file} {option} {update_file}` |                                                              |
| 23     | test_gbpt_Newupdate                                          | `./bpt_test {insert_file} {search_file} {option} {update_file}` |                                                              |
| 38     | test_gbpt_Newupdate & multi_prefix_sort_1thread_noleaf       | `./bpt_test {insert_file} {search_file} {option} {update_file}` |                                                              |
| 33     | bit_multi_prefix_sort_8thread                                | `./bpt_test {insert_file} {search_file} {option} {start_bit} {end_bit: 64 or 32}` | start bit must be less than end_bit, and end_bit must be 64 or 32 |
| 34     | bit_multi_prefix_sort_1thread                                | `./bpt_test {insert_file} {search_file} {option} {start_bit} {end_bit: 64 or 32}` | start bit must be less than end_bit, and end_bit must be 64 or 32 |
| 37     | PPI_BPT_Search_GPU_V2_8thread                                | `./bpt_test {insert_file} {search_file} {option} {start_bit} {end_bit: 64 or 32}` | start bit must be less than end_bit, and end_bit must be 64 or 32 |
| 39     | test_gbpt_Newupdate_batch                                    | `./bpt_test {insert_file} {search_file} {option} {update_file} {insert_file_lines} {update_file_lines} ` | update files may repeat several times, `update_file_lines = repeat * update_file_lines `, see `update-rebuild-ppi-test-batch-repeat.sh` |
| 40     | test_gbpt_update_batch                                       | `./bpt_test {insert_file} {search_file} {option} {update_file} {insert_file_lines}  ` | see `tests/update-test-repeat2.sh`                           |
| 2      | test_prefix_sort_8thread                                     | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 3      | test_sort_8thread                                            | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 4      | PPI_BPT_Search_GPU_V9_thread_scheduling                      | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 7      | test_sort_2thread                                            | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 8      | PPI_BPT_Search_GPU_V14_1_prefix_up_3_level_constant_down_readonly | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 9      | PPI_BPT_Search_GPU_V14_2_prefix_up_full_constant_down_readonly | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 10     | PPI_BPT_Search_GPU_V10_multigpu_basedv5                      | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 14     | PPI_BPT_Search_GPU_V14_3_prefix_opt                          | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |
| 15     | compare_times_2thread                                        | `./bpt_test {insert_file} {big_search_file} {option}`        | too few search file lines may cause exception                |



