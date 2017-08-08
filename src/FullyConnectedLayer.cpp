#include "FullyConnectedLayer.h"

#include <clBLAS.h>


namespace NN
{
	
	
std::string fullyConnectedLayerCLProgramSrc = ""
"__kernel void copyBiasesKernel(__global float* bias, __global float* output, int layerSize, int layerThickness)"
"{"
"	int base = get_global_id(0);"
"	for(int i = 0; i < layerThickness; i++)"
"		output[i * layerSize + base] = bias[base];"
"	"
"}"
""
""
"__kernel void activationFunctionKernel(__global float* v, float param0, float param1)"
"{"
""
""
"}"
"__kernel void deltaActivationFunctionKernel(__global float* v, float param0, float param1)"
"{"
""
"}"
;
	
	
cl_program FullyConnectedLayer::clProgram = NULL;
cl_kernel FullyConnectedLayer::copyBiasesKernel = NULL;
	
void FullyConnectedLayer::Init()
{
	/* Create kernel program from source file*/
	size_t fullyConnectedLayerCLProgramSrcSize = fullyConnectedLayerCLProgramSrc.length(); 
	cl_int err;
	const char* clProgram = fullyConnectedLayerCLProgramSrc.c_str(); 
	clProgram = clCreateProgramWithSource(clEnvironment->ctx, 1, (const char **)(clProgram), (const size_t *)&fullyConnectedLayerCLProgramSrcSize, &err);	
	err = clBuildProgram(clProgram, 1, &clEnvironment->deviceId, NULL, NULL, NULL);
 
	/* Create data parallel OpenCL copyBiasesKernel */	
	copyBiasesKernel = clCreateKernel(clProgram, "copyBiasesKernel", &err);
}
	
	
FullyConnectedLayer::FullyConnectedLayer()
{
	//set the type
	type = "FullyConnectedLayer";
}


void FullyConnectedLayer::RandomizeWeights(double wmin, double wmax, double bmin, double bmax)
{
	//create random nnubmer generators
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


void FullyConnectedLayer::Allocate()
{
	//allocate memory in opencl device
	cl_int err;
	output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
	biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
	weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
}


void FullyConnectedLayer::ComputeForward()
{	
	//rows=outputs
	//cols=intputs
	
	if(PrevLayer()->type == "FullyConnectedLayer")
	{
		cl_event event = NULL;
		cl_int err;
		
		//get the prev layer 
		FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
		
		//copy biases into the output matrix temporaraly for sgemm computation 
		err = clSetKernelArg(copyBiasesKernel, 0, sizeof(cl_mem), (void *)&biases);
		err = clSetKernelArg(copyBiasesKernel, 1, sizeof(cl_mem), (void *)&output);
		err = clSetKernelArg(copyBiasesKernel, 2, sizeof(int), (void *)&layerSize);
		err = clSetKernelArg(copyBiasesKernel, 3, sizeof(int), (void *)&layerThickness);
		size_t global_item_size = layerSize;
		size_t local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, copyBiasesKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
		
		//Compute weighted input from inputs and weights and biasses z = w * i + b, temmp store in output
		int M = layerSize; //rows of matrix A
		int N = prvL->layerThickness; //cols of matrix B
		int K = inputNumber; //cols of matrix A and rows of matrix B
		int lda = K;
		int ldb = N;
		int ldc = N;
        err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
                          M, N, K,
                          1, weights, 0, lda,
                          prvL->output, 0, ldb, 1,
                          output, 0, ldc,
                          1, &clEnvironment->queue, 0, NULL, &event );
		err = clWaitForEvents(1, &event);
		
		//compute activation function on weighted input
		
		
		//
	}
}


void FullyConnectedLayer::Backpropogate()
{
	
}


}