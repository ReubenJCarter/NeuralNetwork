#pragma once


#include "Util/CommonHeaders.h"
#include "BaseLayer.h"


namespace NN
{


//
//InputLayer Layer, the input vector in host memory is copied to the output cl_mem 
//


class InputLayer: public BaseLayer
{
public:
	int layerSize;
	int layerThickness;
	std::vector<float> input;
	cl_mem output;
	
	InputLayer();
	~InputLayer();
	void SetSize(int layerSize, int layerThickness);
	void ReadOutput(float* buffer);
	virtual void Allocate();
	virtual void ComputeForward();
	virtual void Backpropogate();
}; 


}