#pragma once


#include "Util/CommonHeaders.h"
#include "OpenCLHelper/OpenCLHelper.h"
#include "BaseLayer.h"


namespace NN
{


//
//Fully Connected Layer, every unit is connected to every activation in the previous layer
//


class FullyConnectedLayer: public BaseLayer
{
public:
	static cl_program clProgram;
	static cl_kernel copyBiasesKernel;	
	static cl_kernel activationIdentityKernel;	
	static cl_kernel deltaActivationIdentityKernel;	
	static cl_kernel activationBinaryStepKernel;	
	static cl_kernel deltaActivationBinaryStepKernel;	
	static cl_kernel activationLogisticKernel;	
	static cl_kernel deltaActivationLogisticKernel;	

	
	static void Init();
	
public:
	enum ACTIVATION_TYPE{IDENTITY, BINARY_STEP, LOGISTIC};
	ACTIVATION_TYPE activationType; 
	int layerSize;
	int inputNumber;
	int layerThickness;
	cl_mem output;
	cl_mem error;
	cl_mem weights;
	cl_mem biases;

	FullyConnectedLayer();
	~FullyConnectedLayer();
	void SetSize(int lSize);
	void RandomizeWeights(double wmin, double wmax, double bmin, double bmax);//need to be allocated before this
	virtual void Allocate();//need to be connected before this 
	virtual void ComputeForward();
	virtual void Backpropogate();
	
	void DebugPrintOutput();
	void DebugPrintError();
	void DebugPrintWeights();
	void DebugPrintBiases();
}; 


}