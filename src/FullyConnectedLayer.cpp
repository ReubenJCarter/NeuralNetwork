#include "FullyConnectedLayer.h"
#include "InputLayer.h"
	
#include "Util/CommonCLSrc.h"
	
#include <clBLAS.h>
	
	
namespace NN
{	
	
	
const std::string fullyConnectedLayerSrc = ""

""
"__kernel void CopyBiasesKernel(__global float* bias, __global float* output, int layerSize, int layerThickness)"
"{"	
"	int inx = get_global_id(0);"
"	for(int i = 0; i < layerThickness; i++)"
"		output[i * layerSize + inx] = bias[inx];"
"	"
"}"	
""
"__kernel void ActivationKernel(__global float* v, __global float* y, float param0, float param1, ACTIVATION_TYPE activationType)"
"{"
"	int inx = get_global_id(0);"
"	float x = ActivationFunction(v[inx], param0, param1, activationType);"
"	y[inx] = x;"
"}"
""
""
"__kernel void LastLayerErrorKernel(__global float* error, __global float* activation, __global float* trainExamples, __global float* weightedSum, COST_TYPE costType, ACTIVATION_TYPE activationType)"
"{"	
"	int inx = get_global_id(0);"
"	float deltaCost = DeltaCostFunction(activation[inx], trainExamples[inx], costType);"
"	float deltaActivation = DeltaActivationFunction(weightedSum[inx], 0, 0, activationType);"
"	error[inx] = deltaCost * deltaActivation;"
"}"	
""
""
"__kernel void BackpropogateKernel(__global float* error, __global float* nextLayerWeightsTransMultByError, __global float* weightedSum, ACTIVATION_TYPE activationType)"
"{"
"	int inx = get_global_id(0);"
"	error[inx] = nextLayerWeightsTransMultByError[inx] * DeltaActivationFunction(weightedSum[inx], 0, 0, activationType);"
"}"
""
""
"__kernel void ReduceRowsKernel(__global float* bias, __global float* error, float learningRate, int layerSize, int layerThickness)"
"{"	
"	int inx = get_global_id(0);"
"	float reducedVal = 0;"
"	for(int i = 0; i < layerThickness; i++)"
"		reducedVal += error[i * layerSize + inx];"
"	bias[inx] -= learningRate * reducedVal;"
"}"	
;
		
cl_program FullyConnectedLayer::clProgram = NULL;
	
//ActivationFunction
cl_kernel FullyConnectedLayer::activationKernel = NULL;
		
//Copy Biases
cl_kernel FullyConnectedLayer::copyBiasesKernel = NULL;

//lastLayerErrorKernel
cl_kernel FullyConnectedLayer::lastLayerErrorKernel = NULL; 

//backpropogateKernel
cl_kernel FullyConnectedLayer::backpropogateKernel = NULL;

//reduceRowsKernel
cl_kernel FullyConnectedLayer::reduceRowsKernel = NULL;
	
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
	activationKernel = clCreateKernel(clProgram, "ActivationKernel", &err);
	
	//copy bias kernel
	copyBiasesKernel = clCreateKernel(clProgram, "CopyBiasesKernel", &err);
	
	//last layer kernel
	lastLayerErrorKernel = clCreateKernel(clProgram, "LastLayerErrorKernel", &err);
	
	//backpropogate kernel
	backpropogateKernel = clCreateKernel(clProgram, "BackpropogateKernel", &err);
	
	//reduceRows kernel
	reduceRowsKernel = clCreateKernel(clProgram, "ReduceRowsKernel", &err);
}	
	
	
FullyConnectedLayer::FullyConnectedLayer()
{	
	//set the type
	type = "FullyConnectedLayer";
	
	//set default activation function
	activationType = LOGISTIC; 
	
	//set default cost function type
	costType = QUADRATIC;
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


void FullyConnectedLayer::ComputeOutputError(float* trainExamplesBuffer)
{
	cl_event event = NULL;
	cl_int err;
	
	//We presume a forward pass has already happened
	
	//create cl memory for the train exemple
	cl_mem trainExamples = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
	err = clEnqueueWriteBuffer(clEnvironment->queue, trainExamples, CL_TRUE, 0, layerSize * layerThickness * sizeof(float), trainExamplesBuffer, 0, NULL, NULL);
	
	//run the last layer error kernel
	err = clSetKernelArg(lastLayerErrorKernel, 0, sizeof(cl_mem), (void *)&error);
	err = clSetKernelArg(lastLayerErrorKernel, 1, sizeof(cl_mem), (void *)&output);
	err = clSetKernelArg(lastLayerErrorKernel, 2, sizeof(cl_mem), (void *)&trainExamples);
	err = clSetKernelArg(lastLayerErrorKernel, 3, sizeof(cl_mem), (void *)&weightedSum);
	err = clSetKernelArg(lastLayerErrorKernel, 4, sizeof(int), (void *)&costType);
	err = clSetKernelArg(lastLayerErrorKernel, 5, sizeof(int), (void *)&activationType);
	size_t global_item_size = layerSize * layerThickness;
	size_t local_item_size = 1;
	err = clEnqueueNDRangeKernel(clEnvironment->queue, lastLayerErrorKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
	err = clWaitForEvents(1, &event);
	
	//Release trainExampel memory
	clReleaseMemObject(trainExamples);
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
		err = clSetKernelArg(activationKernel, 0, sizeof(cl_mem), (void *)&weightedSum);
		err = clSetKernelArg(activationKernel, 1, sizeof(cl_mem), (void *)&output);
		err = clSetKernelArg(activationKernel, 2, sizeof(float), &activationParam0);
		err = clSetKernelArg(activationKernel, 3, sizeof(float), &activationParam1);
		err = clSetKernelArg(activationKernel, 4, sizeof(int), &activationType);
		global_item_size = layerSize * layerThickness;
		local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, activationKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
		
		//
	}
}

void FullyConnectedLayer::Backpropogate()
{
	if(NextLayer()->type == "FullyConnectedLayer")
	{
		FullyConnectedLayer* nxtL = (FullyConnectedLayer*)NextLayer();
		
		cl_event event = NULL;
		cl_int err;
		
		//compute the transpose of the next layers weights matrix multiplied by the next layers error
		int M = layerSize; //rows of matrix A
		int N = layerThickness; //cols of matrix B
		int K = inputNumber; //cols of matrix A and rows of matrix B
		int lda = M;
		int ldb = K;
		int ldc = M;
        err = clblasSgemm(clblasColumnMajor, clblasTrans, clblasNoTrans, ///GET RID OF MULTIPLY CY C ON THIS
                          M, N, K,
                          1, nxtL->weights, 0, lda,
                          nxtL->error, 0, ldb, 0,
                          error, 0, ldc,
                          1, &clEnvironment->queue, 0, NULL, &event );//column major order gemm, multiply input by weights and adds biases in one step
		err = clWaitForEvents(1, &event);
		
		//hadamard product between the result and deltaActivation(weightedSum (this layers) )
		err = clSetKernelArg(backpropogateKernel, 0, sizeof(cl_mem), (void *)&error);
		err = clSetKernelArg(backpropogateKernel, 1, sizeof(cl_mem), (void *)&error);
		err = clSetKernelArg(backpropogateKernel, 2, sizeof(cl_mem), (void *)&weightedSum);
		err = clSetKernelArg(backpropogateKernel, 3, sizeof(int), &activationType);
		size_t global_item_size = layerSize * layerThickness;
		size_t local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, backpropogateKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
	}
}	

void FullyConnectedLayer::AdjustWeightsBiases()
{
	if(PrevLayer()->type == "FullyConnectedLayer")
	{
		FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
		
		cl_event event = NULL;
		cl_int err;
		float learningRate = 0.01;
		
		//multiply error by transposed activation of the previous layer subtract learningRate fraction from weights 
		int M = layerSize; //rows of matrix A
		int N = layerThickness; //cols of matrix B
		int K = inputNumber; //cols of matrix A and rows of matrix B
		int lda = M;
		int ldb = K;
		int ldc = M;
        err = clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasTrans, //USE INCREMENT C TO DO THE DELTA ADD TO WEIGHTS BY LEARNING RATE
                          M, N, K,
                          -learningRate, error, 0, lda,
                          prvL->output, 0, ldb, 1,
                          weights, 0, ldc,
                          1, &clEnvironment->queue, 0, NULL, &event );//column major order gemm, multiply input by weights and adds biases in one step
		err = clWaitForEvents(1, &event);
		
		//sum the errors for each training example, sum errors across the thickness (reduce the row) (sum up to make nx1 matrix), and multiply by learning rate
		err = clSetKernelArg(reduceRowsKernel, 0, sizeof(cl_mem), (void *)&biases);
		err = clSetKernelArg(reduceRowsKernel, 1, sizeof(cl_mem), (void *)&error);
		err = clSetKernelArg(reduceRowsKernel, 2, sizeof(float), (void *)&learningRate);
		err = clSetKernelArg(reduceRowsKernel, 3, sizeof(int), &layerSize);
		err = clSetKernelArg(reduceRowsKernel, 4, sizeof(int), &layerThickness);
		size_t global_item_size = layerSize;
		size_t local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, reduceRowsKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
	}
}
	
	
}	