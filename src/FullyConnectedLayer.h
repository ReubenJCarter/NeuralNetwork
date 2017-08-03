#pragma once


#include "Util/CommonHeaders.h"
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
	std::vector<float> weights;

	virtual void Allocate(int layerSz);
	virtual void ComputeForward();
	virtual void Backpropogate();
}; 


}