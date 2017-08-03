#include "BaseLayer.h"


namespace NN
{
	

bool BaseLayer::ConnectOutput(BaseLayer* nxt)
{
	bool canConnect = true;
	for(int i = 0; i < nextL.size(); i++)
	{
		if(nextL[i] == nxt)
		{
			canConnect = false;
			std::cerr << "BaseLayer:ConnectOutput():" << "could not connect output as alread connected" << std::endl;
			break;
		}
	}
	if(!canConnect)
	{
		nextL.push_back(nxt);
		nxt->prevL.push_back(this);
	}
	return canConnect; 
}


bool BaseLayer::ConnectInput(BaseLayer* prv)
{
	bool canConnect = true;
	for(int i = 0; i < prevL.size(); i++)
	{
		if(prevL[i] == prv)
		{
			canConnect = false;
			std::cerr << "BaseLayer:ConnectInput():" << "could not connect input as alread connected" << std::endl;
			break;
		}
	}
	if(!canConnect)
	{
		prevL.push_back(prv);
		prv->nextL.push_back(this);
	}
	return canConnect; 
}


BaseLayer* BaseLayer::NextLayer(int inx)
{
	if(inx < nextL.size())
	{
		return nextL[inx];
	}
	else
	{
		std::cerr << "BaseLayer:NextLayer():" << "could not get Next layer, index out of range" << std::endl;
		return NULL; 
	}
}


BaseLayer* BaseLayer::PrevLayer(int inx)
{
	if(inx < prevL.size())
	{
		return prevL[inx];
	}
	else
	{
		std::cerr << "BaseLayer:PrevLayer():" << "could not get Prev layer, index out of range" << std::endl;
		return NULL; 
	}
}
	
	
}