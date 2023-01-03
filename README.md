# Harmonia

Harmonia is a high performance GPU B+Tree implementation. Our work has been posted on [PPoPP'19](https://dl.acm.org/doi/10.1145/3293883.3295704). This repository contains all source codes of Harmonia, and several shell/python scripts are also included to replay our results quickly. Please feel free to make improvements and propose issues.

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

## Steps

1. `cd ./harmonia`
2. Build docker image and start a docker container `sh startup.sh`. This will take some minutes to prepare a necessary environment for Harmonia, please hold on.

3. After a successful step 2, you may see something like below by running `docker images` and `docker ps -a`.

  * ```
  	REPOSITORY    TAG                        IMAGE ID       CREATED         SIZE
  	harmonia      cuda-10.0                  b18895b2ceb7   19 hours ago    5.24GB
  	```

  * ```
  	CONTAINER ID   IMAGE                                  COMMAND                 CREATED        STATUS                   PORTS                                     NAMES
  	34e9fcc4c8d9   harmonia:cuda-10.0                     "/sbin/entrypoint.sh"   19 hours ago   Up 19 hours              0.0.0.0:14722->22/tcp, :::14722->22/tcp   funny_dijkstra
  	```

4. Now you can enter the Harmonia container by running `ssh -p14722 root@localhost` or `ssh -p14722 root@{ip address}`, the root user password is `root`.

5.  After ssh connected, `cd /harmonia/` will show the files and directories for Harmonia.

6.  You can run a simple tests by `cd /harmonia/open-harmonia/ && sh test.sh`

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





