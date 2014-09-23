set(appSrc
	ffnnTool.cpp
)
foreach(iAppSrc ${appSrc})
	add_executable(${iAppSrc} ${iAppSrc})
	target_link_libraries(${iAppSrc} ai)
endforeach(iAppSrc)