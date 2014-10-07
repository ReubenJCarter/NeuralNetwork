set(aiSrc
	matd.cpp
	FFNN.cpp
	fileAccess/csvFileParser.cpp
	fileAccess/matrixLoader.cpp
	fileAccess/matrixSaver.cpp
)	
add_library(ai STATIC ${aiSrc})
if(use_cblas)
	target_link_libraries(ai openblas)
	message(STATUS "Compiling with feature cblas")
	add_definitions(-DCBLAS)
endif()
