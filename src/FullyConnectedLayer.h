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
	int layerSize;
	int inputNumber;
	cl_mem output;
	cl_mem error;
	cl_mem weights;
	cl_mem biases;

	FullyConnectedLayer();
	void RandomizeWeights(double wmin, double wmax, double bmin, double bmax);//need to be allocated before this
	virtual void Allocate();//need to be connected before this 
	virtual void ComputeForward();
	virtual void Backpropogate();
}; 


}