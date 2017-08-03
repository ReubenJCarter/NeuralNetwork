set(NNSrc
	Util/LoadImage.cpp
	Util/SaveImage.cpp
	Util/MNISTImageFile.cpp
	Util/MNISTLabelFile.cpp
	BaseLayer.cpp
	FullyConnectedLayer.cpp
)	

add_library(NN STATIC ${NNSrc})

target_link_libraries(NN Devil)