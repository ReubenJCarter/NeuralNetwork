#include "FullyConnectedLayer.h"
#include "InputLayer.h"
	
#include "Util/CommonCLSrc.h"
	
#include <clBLAS.h>
	
	
namespace NN
{	
	
	
const std::string fullyConnectedLayerSrc = ""
"typedef enum {IDENTITY, BINARY_STEP, LOGISTIC} ACTIVATION_TYPE;"
""
"__kernel void CopyBiases(__global float* bias, __global float* output, int layerSize, int layerThickness)"
"{"	
"	int base = get_global_id(0);"
"	for(int i = 0; i < layerThickness; i++)"
"		output[i * layerSize + base] = bias[base];"
"	"
"}"	
""
"void ActivationFunction(__global float* v, __global float* y, float param0, float param1, ACTIVATION_TYPE activationType)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	y[base] = x;"
"}"
""
"__kernel void DeltaQuatraticCostFunction(__global float* output, __global float* activation, __global float* trainExample)"
"{"	
"	int i = get_global_id(0);"
"	output[i] = activation[i] - trainExample[i];"
"}"	
""
"__kernel void LastLayerError(__global float* error, __global float* activation, __global float* trainExample, __global float* weightedSum, int costFunction, ACTIVATION_TYPE activationType)"
"{"	
"	int i = get_global_id(0);"
"	float deltaCost;" 
"	if(costFunction == 0)"
"		deltaCost = activation[i] - trainExample[i];"

"	float deltaActivation;"
"	"
"}"	
;
	
	
cl_program FullyConnectedLayer::clProgram = NULL;
	
//ActivationFunction
cl_kernel FullyConnectedLayer::activationFunctionKernel = NULL;
		
//Copy Biases
cl_kernel FullyConnectedLayer::copyBiasesKernel = NULL;
	
void FullyConnectedLayer::Init()
{	
	//build src
	std::string fullyConnectedLayerCLProgramSrc = "";
	fullyConnectedLayerCLProgramSrc += activationFunctionsSrc;
	fullyConnectedLayerCLProgramSrc += fullyConnectedLayerSrc;
	
	/* Create kernel program from source file*/
	size_t fullyConnectedLayerCLProgramSrcSize = fullyConnectedLayerCLProgramSrc.length(); 
	cl_int err;
	const char* clProgramSrc = fullyConnectedLayerCLProgramSrc.c_str(); 
	clProgram = clCreateProgramWithSource(clEnvironment->ctx, 1, (const char **)(&clProgramSrc), (const size_t *)&fullyConnectedLayerCLProgramSrcSize, &err);	
	err = clBuildProgram(clProgram, 1, &clEnvironment->deviceId, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) 
	{
		// Determine the size of the log
		size_t logsize;
		clGetProgramBuildInfo(clProgram, clEnvironment->deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);

		// Allocate memory for the log
		std::vector<char> logBuffer(logsize);

		// Get the log
		clGetProgramBuildInfo(clProgram, clEnvironment->deviceId, CL_PROGRAM_BUILD_LOG, logsize, &logBuffer[0], NULL);
		std::string log(logBuffer.begin(), logBuffer.end());
		
		// Print the log
		std::cout << log;
	}
	
	/* Create data parallel OpenCL kernels*/	
	//create activation kernels
	
	//ActivationFunction
	activationFunctionKernel = clCreateKernel(clProgram, "ActivationFunction", &err);
	
	//copy bias kernel
	copyBiasesKernel = clCreateKernel(clProgram, "CopyBiases", &err);
	
	
	
}	
	
	
FullyConnectedLayer::FullyConnectedLayer()
{	
	//set the type
	type = "FullyConnectedLayer";
	
	//set default activation function
	activationType = LOGISTIC; 
}	
	
FullyConnectedLayer::~FullyConnectedLayer()
{	
	if(isMemoryAllocated)
	{
		//release cl memory 
		clReleaseMemObject(weightedSum);
		clReleaseMemObject(output);
		clReleaseMemObject(biases);
		clReleaseMemObject(weights);
	}
}	
	
void FullyConnectedLayer::SetSize(int lSize)
{	
	layerSize = lSize; 
}	
	
void FullyConnectedLayer::RandomizeWeights(double wmin, double wmax, double bmin, double bmax)
{	
	//create random nubmer generators
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine randGenerator(seed);
	std::normal_distribution<float> randWeight(wmin, wmax);
	std::normal_distribution<float> randBias(bmin, bmax);
	std::vector<float> weightBuf(layerSize * inputNumber);
	std::vector<float> biasBuf(layerSize);
	
	//Generate all the random weights as biases
	for(unsigned int i = 0; i < layerSize * inputNumber; i++)
	{
		weightBuf[i] = randWeight(randGenerator);
	}
	for(unsigned int i = 0; i < layerSize; i++)
	{
		biasBuf[i] = randBias(randGenerator);
	}
	
	//write the random weights and biases to the openCL device memory
	cl_int err;
	err = clEnqueueWriteBuffer(clEnvironment->queue, weights, CL_TRUE, 0, layerSize * inputNumber * sizeof(float), &weightBuf[0], 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clEnvironment->queue, biases, CL_TRUE, 0, layerSize * sizeof(float), &biasBuf[0], 0, NULL, NULL);
}	
	
	
void FullyConnectedLayer::ReadOutput(float* buffer)
{	
	//read the output buffer from opencl
	cl_int err;
	err = clEnqueueReadBuffer(clEnvironment->queue, output, CL_TRUE, 0, layerSize * layerThickness * sizeof(float), buffer, 0, NULL, NULL);
}	


void FullyConnectedLayer::ComputeOutputError(float* outputValues)
{
	//We presume a forward pass has already happened
	
	//compute derivative of activation function with zL
	
	
	//compute the derivative of the cost fucntion with aL (the activation)
	
	
	//compute hadamard product of deriv cost and deriv activation
	
	
}
	
	
void FullyConnectedLayer::Allocate()
{	
	if(PrevLayer()->type == "FullyConnectedLayer")
	{
		FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
		inputNumber = prvL->layerSize;
		layerThickness = prvL->layerThickness; // this can be set from the input layer
	
		//allocate memory in opencl device
		cl_int err;
		weightedSum = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
		weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
		
		isMemoryAllocated = true; 
	}
	else if(PrevLayer()->type == "InputLayer")
	{
		InputLayer* prvL = (InputLayer*)PrevLayer();
		inputNumber = prvL->layerSize;
		layerThickness = prvL->layerThickness; // this can be set from the input layer
	
		//allocate memory in opencl device
		cl_int err;
		weightedSum = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
		weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
		
		isMemoryAllocated = true; 
	}
}	
	
	
void FullyConnectedLayer::ComputeForward()
{	
	if(!isMemoryAllocated)
		Allocate();

	
	if(PrevLayer()->type == "FullyConnectedLayer" || PrevLayer()->type == "InputLayer")
	{	
		cl_event event = NULL;
		cl_int err;
		
		
		//get the prev layer output (the input )
		cl_mem input;
		if(PrevLayer()->type == "FullyConnectedLayer")
		{
			FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
			input = prvL->output;
		}
		else if(PrevLayer()->type == "InputLayer")
		{
			InputLayer* prvL = (InputLayer*)PrevLayer();
			input = prvL->output;
		}
		
		//copy biases into the weightedSum matrix temporaraly for sgemm computation 
		err = clSetKernelArg(copyBiasesKernel, 0, sizeof(cl_mem), (void *)&biases);
		err = clSetKernelArg(copyBiasesKernel, 1, sizeof(cl_mem), (void *)&weightedSum);
		err = clSetKernelArg(copyBiasesKernel, 2, sizeof(int), &layerSize);
		err = clSetKernelArg(copyBiasesKernel, 3, sizeof(int), &layerThickness);
		size_t global_item_size = layerSize;
		size_t local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, copyBiasesKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
		
		//Compute weighted input from inputs and weights and biasses z = w * i + b, temmp store in output
		int M = layerSize; //rows of matrix A
		int N = layerThickness; //cols of matrix B
		int K = inputNumber; //cols of matrix A and rows of matrix B
		int lda = M;
		int ldb = K;
		int ldc = M;
        err = clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans,
                          M, N, K,
                          1, weights, 0, lda,
                          input, 0, ldb, 1,
                          weightedSum, 0, ldc,
                          1, &clEnvironment->queue, 0, NULL, &event );//column major order gemm, multiply input by weights and adds biases in one step
		err = clWaitForEvents(1, &event);
		
		//compute activation function on weighted input
		err = clSetKernelArg(activationFunctionKernel, 0, sizeof(cl_mem), (void *)&weightedSum);
		err = clSetKernelArg(activationFunctionKernel, 1, sizeof(cl_mem), (void *)&output);
		err = clSetKernelArg(activationFunctionKernel, 2, sizeof(float), &activationParam0);
		err = clSetKernelArg(activationFunctionKernel, 3, sizeof(float), &activationParam1);
		err = clSetKernelArg(activationFunctionKernel, 4, sizeof(int), &activationType);
		global_item_size = layerSize * layerThickness;
		local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, activationFunctionKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
		
		//
	}
}	
	
	
void FullyConnectedLayer::Backpropogate()
{
	
}
	
	
}	