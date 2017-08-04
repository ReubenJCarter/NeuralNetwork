cmake_minimum_required(VERSION 2.6)
project(NeuralNetork)
set(PROJECT_BRIEF "NeuralNetork")
set(CMAKE_BUILD_TYPE RELEASE)
set(GCC_COMPILE_FLAGS "-std=c++11 -Wall -O3 -g")
set(GCC_LINK_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_LINK_FLAGS}")


#Devil
set(DEVIL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/devil/include  CACHE PATH  "path to Devil include directory")
set(DEVIL_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/devil/lib/x64/Release  CACHE PATH  "path to Devil include directory")
include_directories(${DEVIL_INC_DIR})
link_directories(${DEVIL_LIB_DIR})


#clBLAS
set(CLBLAS_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/clBLAS-2.12.0-Windows-x64/package/include  CACHE PATH  "path to clBLAS include directory")
set(CLBLAS_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/clBLAS-2.12.0-Windows-x64/package/lib64/import  CACHE PATH  "path to clBLAS include directory")
include_directories(${CLBLAS_INC_DIR})
link_directories(${CLBLAS_LIB_DIR})


#opencl
set(OPENCL_INC_DIR "C:\\Program Files (x86)\\AMD APP SDK\\3.0\\include" CACHE PATH  "path to opencl include directory")
set(OPENCL_LIB_DIR "C:\\Program Files (x86)\\AMD APP SDK\\3.0\\lib\\x86_64" CACHE PATH "path to opencl lib directory")
include_directories(${OPENCL_INC_DIR})
link_directories(${OPENCL_LIB_DIR})


include_directories(src)
add_subdirectory(src)
add_subdirectory(test)