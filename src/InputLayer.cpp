#include "InputLayer.h"


namespace NN
{


//
//InputLayer Layer, the input vector in host memory is copied to the output cl_mem 
//

InputLayer::InputLayer()
{
	type = "InputLayer";
}

InputLayer::~InputLayer()
{
	if(isMemoryAllocated)
	{
		clReleaseMemObject(output);
	}
}

void InputLayer::SetSize(int lSize, int lThickness)
{
	layerSize = lSize; 
	layerThickness = lThickness;
}

void InputLayer::SetInput(float* inp)
{
	input = inp;
}

void InputLayer::ReadOutput(float* buffer)
{	
	//read the output buffer from opencl
	cl_int err;
	err = clEnqueueReadBuffer(clEnvironment->queue, output, CL_TRUE, 0, layerSize * layerThickness * sizeof(float), buffer, 0, NULL, NULL);
}	

void InputLayer::Allocate()
{
	//allocate memory in opencl device
	cl_int err;
	output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
	
	isMemoryAllocated = true;
}

void InputLayer::ComputeForward()
{
	//make sure memory is allocated to compute forwards
	if(!isMemoryAllocated)
		Allocate();
	
	//copy host input data to cl mem output
	cl_event event = NULL;
	cl_int err;
	clEnqueueWriteBuffer(clEnvironment->queue, output, CL_FALSE, 0, layerSize * layerThickness * sizeof(float), input, 0, NULL, &event);
	err = clWaitForEvents(1, &event);
}

void InputLayer::Backpropogate()
{
	
}


}