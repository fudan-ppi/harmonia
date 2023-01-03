cmake_minimum_required(VERSION 3.5.0)

project(BPlussTree-Acc)

include(FindCUDA)
find_package(CUDA REQUIRED)

# set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
#find_package(Boost 1.62.0 COMPONENTS system thread)
#find_package(Boost)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -mavx -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -lboost_system -lboost_thread -mavx -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -lboost_system -lboost_thread -mavx -mavx2 -fopenmp")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -lpapi -mavx -mavx2 -fopenmp")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    #message("cuda debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -gencode arch=compute_70,code=sm_70 -O0 -g -G")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_61,code=sm_61 -O0 -g -G")
else()
    #message("cuda release")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -gencode arch=compute_70,code=sm_70 -O3")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_61,code=sm_61 -O3")
endif()

include_directories(./src ./cub)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARIES_DIRS})
link_directories(./lib)

aux_source_directory(./src SOURCE_LIST)
file(GLOB GPU_SOURCE_LIST "src/*.cu")
file(GLOB REGULAR_GPU_LIST "regular_bpt/*.cu")

aux_source_directory(./regular_bpt REGULAR_TREE_SOURCE_LIST)

file(GLOB_RECURSE REMOVE_CMAKE "src/data_sort.cu")
list(REMOVE_ITEM GPU_SOURCE_LIST  ${REMOVE_CMAKE})
list(REMOVE_ITEM SOURCE_LIST  ${REMOVE_CMAKE})


cuda_add_executable(bpt_test ${SOURCE_LIST} ${GPU_SOURCE_LIST})
target_link_libraries(bpt_test libpapi.a ${Boost_LIBRARIES}) 

cuda_add_executable(generate-partially-sort-dataset ./src/data_sort.cu)


add_definitions(-DDEFAULT_ORDER=${ORDER}) #8, 16 ,32 , 64, 128
add_definitions(-DNarrow_Rate=${NARROW}) #1, 2, 4, 8
cuda_add_executable(regular_bpt_test ${REGULAR_TREE_SOURCE_LIST} ${REGULAR_GPU_LIST})
 

