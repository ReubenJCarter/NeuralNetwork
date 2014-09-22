set(testSrc
	csvFileTest.cpp
	matdTest.cpp
	FFNNTest.cpp
)
foreach(iTestSrc ${testSrc})
	add_executable(${iTestSrc} ${iTestSrc})
	target_link_libraries(${iTestSrc} ai)
endforeach(iTestSrc)