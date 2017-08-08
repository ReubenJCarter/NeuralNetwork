#OpenCL startup test
add_executable(OpenclStartupTest OpenclStartupTest.cpp)
target_link_libraries(OpenclStartupTest OpenCL)


#clBLAS startup test
add_executable(CLBLASStartupTest CLBLASStartupTest.cpp)
target_link_libraries(CLBLASStartupTest OpenCL clBLAS)


#MNIST hand written digits test
add_executable(MNISTTest MNISTTest.cpp)
target_link_libraries(MNISTTest NN)

#Fully Connected Layer Test
add_executable(FullyConnectedLayerTest FullyConnectedLayerTest.cpp)
target_link_libraries(FullyConnectedLayerTest NN)

