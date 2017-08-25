#pragma once


#include "Util/CommonHeaders.h"
#include "OpenCLHelper/OpenCLHelper.h"
#include "BaseLayer.h"


namespace NN
{


//
//Fully Connected Layer, every unit is connected to every activation in the previous layer, everything is done column major order
//


class FullyConnectedLayer: public BaseLayer
{
public:
	static cl_program clProgram;
	static cl_kernel copyBiasesKernel;	
	static cl_kernel activationKernel;	
	static cl_kernel lastLayerErrorKernel;		
	static cl_kernel backpropogateKernel;
	static cl_kernel reduceRowsKernel;

	
	static void Init();
	
public:
	enum ACTIVATION_TYPE{IDENTITY, BINARY_STEP, LOGISTIC};
	enum COST_TYPE{QUADRATIC};
	enum RANDOMIZATION{RAND_NORM_DIST}; 
	ACTIVATION_TYPE activationType; 
	COST_TYPE costType;
	int layerSize;
	int inputNumber;
	int layerThickness;
	float learningRate;
	cl_mem weightedSum;
	cl_mem output;
	cl_mem error;
	cl_mem weights;
	cl_mem biases;
	
	float activationParam0;
	float activationParam1;

	FullyConnectedLayer();
	~FullyConnectedLayer();
	void SetSize(int lSize);
	void RandomizeWeights(double wmin, double wmax, double bmin, double bmax);//need to be allocated before this
	void ReadOutput(float* buffer);
	void ReadError(float* buffer);
	void ComputeOutputError(float* trainExamplesBuffer);
	virtual void Allocate();//need to be connected before this 
	virtual void ComputeForward();
	virtual void Backpropogate();
	void AdjustWeightsBiases();
}; 


}