cmake_minimum_required(VERSION 2.6)
project(NeuralNetork)
set(PROJECT_BRIEF "NeuralNetork")
set(CMAKE_BUILD_TYPE RELEASE)
set(GCC_COMPILE_FLAGS "-std=c++11 -Wall -O3 -g")
set(GCC_LINK_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_LINK_FLAGS}")


#Devil
set(DEVIL_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/devil/include)
set(DEVIL_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/devil/lib/x64/Release)
include_directories(${DEVIL_INC_DIR})
link_directories(${DEVIL_LIB_DIR})


include_directories(src)
add_subdirectory(src)
add_subdirectory(test)