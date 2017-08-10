#include "FullyConnectedLayer.h"
#include "InputLayer.h"

#include <clBLAS.h>


namespace NN
{
	
//
//Activation Function 
//

const std::string activationFunctionsSrc = ""
"__kernel void ActivationIdentity(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x;"
"}"
"__kernel void DeltaActivationIdentity(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = 1;"
"}"
""
"__kernel void ActivationBinaryStep(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? 0 : 1;"
"}"
"__kernel void DeltaActivationBinaryStep(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = 0;"
"}"
""
"__kernel void ActivationLogistic(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = 1.0 /(1.0 + exp(-x));"
"}"
"__kernel void DeltaActivationLogistic(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	float f = 1.0 /(1.0 + exp(-x));"
"	v[base] = f * (1.0 - f);"
"}"
""
"__kernel void ActivationTanH(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = tanh(x);"
"}"
"__kernel void DeltaActivationTanH(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	float f = tanh(x);"
"	v[base] = 1.0 - f * f;"
"}"
""
"__kernel void ActivationArcTan(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = atan(x);"
"}"
"__kernel void DeltaActivationArcTan(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = 1.0/(x * x + 1);"
"}"
""
"__kernel void ActivationSoftSign(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x / (1 + abs(x));"
"}"
"__kernel void DeltaActivationSoftSign(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	float f = 1 + abs(x);"
"	v[base] = 1.0 / (t * t);"
"}"
""
"__kernel void ActivationReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? 0 : x;"
"}"
"__kernel void DeltaActivationReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? 0 : 1;"
"}"
""
"__kernel void ActivationLeakyReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? 0.01 * x : x;"
"}"
"__kernel void DeltaActivationLeakyReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? 0.01 : 1;"
"}"
""
"__kernel void ActivationLeakyPReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? param0 * x : x;"
"}"
"__kernel void DeltaActivationLeakyPReLU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? param0 : 1;"
"}"
""
"__kernel void ActivationELU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = x < 0 ? param0 * (exp(x) - 1.0) : x;"
"}"
"__kernel void DeltaActivationELU(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	float f = param0 * (exp(x) - 1.0);"
"	v[base] = x < 0 ? f + param0 : 1;"
"}"
""
"__kernel void ActivationSoftPlus(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = log(1.0 + exp(x));"
"}"
"__kernel void DeltaActivationSoftPlus(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = 1.0 / (1.0 + exp(-x));"
"}"
""
"__kernel void ActivationGaussian(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = exp(-x * x);"
"}"
"__kernel void DeltaActivationGaussian(__global float* v, float param0, float param1)"
"{"
"	int base = get_global_id(0);"
"	float x = v[base];"
"	v[base] = -2 * x * exp(-x * x);"
"}"
;
	
const std::string copyBias = ""
"__kernel void CopyBiases(__global float* bias, __global float* output, int layerSize, int layerThickness)"
"{"
"	int base = get_global_id(0);"
"	for(int i = 0; i < layerThickness; i++)"
"		output[i * layerSize + base] = bias[base];"
"	"
"}"
"";
	
	
cl_program FullyConnectedLayer::clProgram = NULL;

//Identity
cl_kernel FullyConnectedLayer::activationIdentityKernel = NULL;
cl_kernel FullyConnectedLayer::deltaActivationIdentityKernel = NULL;

//BinaryStep
cl_kernel FullyConnectedLayer::activationBinaryStepKernel = NULL;
cl_kernel FullyConnectedLayer::deltaActivationBinaryStepKernel = NULL;

//Logistic
cl_kernel FullyConnectedLayer::activationLogisticKernel = NULL;
cl_kernel FullyConnectedLayer::deltaActivationLogisticKernel = NULL;


cl_kernel FullyConnectedLayer::copyBiasesKernel = NULL;
	
void FullyConnectedLayer::Init()
{
	//build src
	std::string fullyConnectedLayerCLProgramSrc = "";
	fullyConnectedLayerCLProgramSrc += activationFunctionsSrc;
	fullyConnectedLayerCLProgramSrc += copyBias;
	
	/* Create kernel program from source file*/
	size_t fullyConnectedLayerCLProgramSrcSize = fullyConnectedLayerCLProgramSrc.length(); 
	cl_int err;
	const char* clProgramSrc = fullyConnectedLayerCLProgramSrc.c_str(); 
	clProgram = clCreateProgramWithSource(clEnvironment->ctx, 1, (const char **)(&clProgramSrc), (const size_t *)&fullyConnectedLayerCLProgramSrcSize, &err);	
	err = clBuildProgram(clProgram, 1, &clEnvironment->deviceId, NULL, NULL, NULL);
 
	/* Create data parallel OpenCL kernels*/	
	//create activation kernels
	
	//Identity
	activationIdentityKernel = clCreateKernel(clProgram, "ActivationIdentity", &err);
	deltaActivationIdentityKernel = clCreateKernel(clProgram, "DeltaActivationIdentity", &err);
	
	//BinaryStep
	activationBinaryStepKernel = clCreateKernel(clProgram, "ActivationBinaryStep", &err);
	deltaActivationBinaryStepKernel = clCreateKernel(clProgram, "DeltaActivationBinaryStep", &err);
	
	//Logistic
	activationLogisticKernel = clCreateKernel(clProgram, "ActivationLogistic", &err);
	deltaActivationLogisticKernel = clCreateKernel(clProgram, "DeltaActivationLogistic", &err);
	
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
	//release cl memory 
	clReleaseMemObject(output);
	clReleaseMemObject(biases);
	clReleaseMemObject(weights);
}

void FullyConnectedLayer::SetSize(int lSize)
{
	layerSize = lSize; 
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
	if(PrevLayer()->type == "FullyConnectedLayer")
	{
		FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
		inputNumber = prvL->layerSize;
		layerThickness = prvL->layerThickness; // this can be set from the input layer
	
		//allocate memory in opencl device
		cl_int err;
		output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
		weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
	}
	else if(PrevLayer()->type == "InputLayer")
	{
		InputLayer* prvL = (InputLayer*)PrevLayer();
		inputNumber = prvL->layerSize;
		layerThickness = prvL->layerThickness; // this can be set from the input layer
	
		//allocate memory in opencl device
		cl_int err;
		output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
		weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
	}
}


void FullyConnectedLayer::ComputeForward()
{	
	//rows=outputs
	//cols=intputs
	
	int prevLayerThickness;
	
	
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
		cl_kernel* activationKernel = &activationLogisticKernel; 
		if(activationType == IDENTITY)
			activationKernel = &activationIdentityKernel;
		else if(activationType == BINARY_STEP)
			activationKernel = &activationBinaryStepKernel; 
		else if(activationType == LOGISTIC)
			activationKernel = &activationLogisticKernel; 
		err = clSetKernelArg(*activationKernel, 0, sizeof(cl_mem), (void *)&output);
		global_item_size = layerSize;
		local_item_size = 1;
		err = clEnqueueNDRangeKernel(clEnvironment->queue, *activationKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		err = clWaitForEvents(1, &event);
		
		//
	}
	else if(PrevLayer()->type == "InputLayer")
	{
		FullyConnectedLayer* prvL = (FullyConnectedLayer*)PrevLayer();
		inputNumber = prvL->layerSize;
		layerThickness = prvL->layerThickness; // this can be set from the input layer
	
		//allocate memory in opencl device
		cl_int err;
		output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * layerThickness * sizeof(float), NULL, &err);
		biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
		weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
	}
}


void FullyConnectedLayer::Backpropogate()
{
	
}


}