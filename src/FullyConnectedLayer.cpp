#include "FullyConnectedLayer.h"


namespace NN
{
	
	
FullyConnectedLayer::FullyConnectedLayer()
{
	type = "FullyConnectedLayer";
}


void FullyConnectedLayer::RandomizeWeights(double wmin, double wmax, double bmin, double bmax)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine randGenerator(seed);
	std::normal_distribution<float> randWeight(wmin, wmax);
	std::normal_distribution<float> randBias(bmin, bmax);
	std::vector<float> weightBuf(layerSize * inputNumber);
	std::vector<float> biasBuf(layerSize);
	for(unsigned int i = 0; i < layerSize * inputNumber; i++)
	{
		weightBuf[i] = randWeight(randGenerator);
	}
	for(unsigned int i = 0; i < layerSize; i++)
	{
		biasBuf[i] = randBias(randGenerator);
	}
	cl_int err;
	err = clEnqueueWriteBuffer(clEnvironment->queue, weights, CL_TRUE, 0, layerSize * inputNumber * sizeof(float), &weightBuf[0], 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clEnvironment->queue, biases, CL_TRUE, 0, layerSize * sizeof(float), &biasBuf[0], 0, NULL, NULL);
}


void FullyConnectedLayer::Allocate()
{
	cl_int err;
	output = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
	biases = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * sizeof(float), NULL, &err);
	weights = clCreateBuffer(clEnvironment->ctx, CL_MEM_READ_WRITE, layerSize * inputNumber * sizeof(float), NULL, &err);
	//err = clEnqueueWriteBuffer(clEnvironment->queue, bufA, CL_TRUE, 0, M * K * sizeof( *A ), A, 0, NULL, NULL );
}


void FullyConnectedLayer::ComputeForward()
{	
	//rows=outputs
	//cols=intputs
	
	if(PrevLayer()->type == "FullyConnectedLayer")
	{
		//Compute output from inputs and weights
		
	}
}


void FullyConnectedLayer::Backpropogate()
{
	
}


}