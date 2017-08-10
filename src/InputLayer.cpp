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
	clReleaseMemObject(output);
}

void InputLayer::SetSize(int lSize, int lThickness)
{
	layerSize = lSize; 
	layerThickness = lThickness;
}

void InputLayer::Allocate()
{
	//allocate memory in opencl device
	cl_int err;
	output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
}

void InputLayer::ComputeForward()
{
	//copy host input data to cl mem output
	cl_event event = NULL;
	cl_int err;
	clEnqueueWriteBuffer(clEnvironment->queue, output, CL_FALSE, 0, input.size() * sizeof(float), &input[0], 0, NULL, &event);
	err = clWaitForEvents(1, &event);
}

void InputLayer::Backpropogate()
{
	
}


}