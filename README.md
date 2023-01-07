# Harmonia: A High Throughput B+tree for GPUs

Harmonia is a high performance GPU B+Tree implementation. This is the source code repository of Harmonia.

Please read the paper ["Harmonia: A High Throughput B+tree for GPUs"](https://dl.acm.org/doi/10.1145/3293883.3295704) of PPoPP'19.

If you use this work, please cite:

```
@inproceedings{10.1145/3293883.3295704,
author = {Yan, Zhaofeng and Lin, Yuzhe and Peng, Lu and Zhang, Weihua},
title = {Harmonia: A High Throughput B+tree for GPUs},
year = {2019},
isbn = {9781450362252},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3293883.3295704},
doi = {10.1145/3293883.3295704},
booktitle = {Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming},
pages = {133–144},
numpages = {12},
keywords = {GPU, high-throughput, B+tree},
location = {Washington, District of Columbia},
series = {PPoPP '19}
}
```

WARNING: The code can be only used for academic research. Please do not use it for production.

Contact information: clzhao16@fudan.edu.cn

## Downloading the source code

```shell
git clone https://github.com/fudan-ppi/harmonia.git
```

## Build Harmonia

1. Build docker image and run docker container
```
cd ./harmonia
sh startup.sh
docker ps -a
```

## Run Harmonia

1. Connect to docker container
```
ssh -p14722 root@localhost
// enter password: root
```

2. Run tests in container
```
cd /harmonia/open-harmonia/ && sh test.sh
```

## Files and Directories

```
├── build.sh                        // build docker image harmonia:cuda-10.0
├── start.sh                        // start a docker container environment for Harmonia
├── Dockerfile                      // Dockerfile to build docker image
├── entrypoint.sh                   // entrypoint of docker container
├── startup.sh                      // build docker image and start a docker container
├── open-harmonia                   // source codes of Harmonia
│   ├── build.sh                    // cmake && make -j
│   ├── clean.sh                    // clean all compile and compiled files
│   ├── cmake_install.cmake         // cmake file
│   ├── CMakeLists.txt              // cmake file
│   ├── compile_commands.json       // compile commands for code editor
│   ├── cub                         // cub library
│   │   └── ...
│   ├── generate.sh                 // generate data files, see script/compile-commands
│   ├── lib                         // libraries
│   │   └── libpapi.a
│   ├── README.md                   // how to compile and run Harmonia
│   ├── regular_bpt                 // source codes
│   │   └── ...
│   ├── script
│   │   ├── compile-commands        // python scripts to generate compile commands
│   │   │   └── ...
│   │   ├── generate-data           // python scripts to generate data files
│   │   │   ├── dataset             // generated data files
│   │   │   │   └── ...
│   │   └── tests                   // a lot of tests shell scripts(CANNOT RUN, see Q&A)
│   │   │   └── ...
│   ├── regular_bpt_test            // executable file to run regular bplus tree tests
│   ├── bpt_test                    // executable file to run Harmonia bplus tree tests
│   ├── generate-partially-sort-dataset   // executable file to generate dataset
│   ├── test.sh                     // a shell script to run several tests
│   └── src                         // source codes
│       └── ...
└── README.md                       // this file
```

## Q&A

* Why changes in container are not reflected on host machine?
  * We use COPY(line 58) in `Dockerfile`, which means all changes in container won't effect your files on host machine.
  * You may modify `Dockerfile` and `start.sh` to mount the host machine directory to the container's inside file system.
* How to run `bpt_test`?
  * Please read the `harmonia/open-harmonia/README.md`.
* Why compile commands don't work?
  * Please check your file path if you have modified the `Dockerfile`.
* Why `harmonia/open-harmonia/script/tests/{xxx}.sh` can not run?
	* Because the original generated data files are located in `harmonia/open-harmonia/script/generate-data/dataset`, but in these shell test scripts, all data files should be located in `harmonia/open-harmonia/data_set`
	* These tests are just examples. If you want to try to run them, you should move the `harmonia/open-harmonia/script/generate-data/dataset` to `harmonia/open-harmonia/data_set`, and then check those file paths in these shell scripts.





