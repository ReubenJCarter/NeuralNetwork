#pragma once


#include "Util/CommonHeaders.h"
#include "OpenCLHelper/OpenCLHelper.h"
#include "BaseLayer.h"


namespace NN
{
	
	
class ConvLayer: public BaseLayer
{
public:
	static cl_program clProgram;
	
	static void Init();
	
public:
	enum ACTIVATION_TYPE{IDENTITY, BINARY_STEP, LOGISTIC};
	enum COST_TYPE{QUADRATIC, CROSS_ENTROPY};
	enum RANDOMIZATION{RAND_NORM_DIST}; 
	ACTIVATION_TYPE activationType; 
	COST_TYPE costType;
	int filterNumber;
	int filterSize;
	int stride; 
	int padding;
	int inputSizeW;
	int inputSizeH;
	int inputSizeD;
	float learningRate;

	ConvLayer();
	~ConvLayer();
	
	void SetFilterSize(int lSize);
	void SetFilterNumber(int fNum);
	void SetStride(int stri);
	void SetPadding(int pad);
	void SetInputSize(int w, int h, int d);
	
	virtual void Allocate();//need to be connected before this 
	
	void RandomizeWeights(double wmin, double wmax, double bmin, double bmax);//need to be allocated before this
	void ComputeOutputError(float* trainExamplesBuffer);
	virtual void ComputeForward();
	virtual void Backpropogate();
	void AdjustWeightsBiases();
	
	void ReadOutput(float* buffer);
	void ReadError(float* buffer);
}; 

	
}