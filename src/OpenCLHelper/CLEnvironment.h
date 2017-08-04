#pragma once 

#include <iostream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <iostream>

namespace CLHelper
{	


class CLEnvironment
{
public:
	cl_context ctx = 0;
    cl_command_queue queue = 0;
	
	CLEnvironment(DeviceInfo& deviceInfo)
	{
		cl_int err;
		cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
		props[1] = (cl_context_properties)deviceInfo.platformId;
		ctx = clCreateContext(props, 1, &(deviceInfo.deviceId), NULL, NULL, &err );
		queue = clCreateCommandQueue(ctx, deviceInfo.deviceId, 0, &err);
	}
};


}