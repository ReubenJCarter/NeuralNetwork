set(aiSrc
	csvFile.cpp
	matd.cpp
	FFNN.cpp
)	
add_library(ai STATIC ${aiSrc})
if(use_cblas)
	target_link_libraries(ai openblas)
	message(STATUS "Compiling with feature cblas")
	add_definitions(-DCBLAS)
endif()
