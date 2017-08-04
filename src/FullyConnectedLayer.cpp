#include "FullyConnectedLayer.h"


namespace NN
{
	
	
FullyConnectedLayer::FullyConnectedLayer()
{
	type = "FullyConnectedLayer";
}


void FullyConnectedLayer::RandomizeWeights(double wmin, double wmax, double bmin, double bmax)
{
	
}


void FullyConnectedLayer::Allocate(int layerSz)
{
	
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