set(aiSrc
	csvFileParser.cpp
	matd.cpp
	FFNN.cpp
	matrixLoader.cpp
	matrixSaver.cpp
)	
add_library(ai STATIC ${aiSrc})
if(use_cblas)
	target_link_libraries(ai openblas)
	message(STATUS "Compiling with feature cblas")
	add_definitions(-DCBLAS)
endif()
