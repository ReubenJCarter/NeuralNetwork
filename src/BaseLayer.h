#pragma once

#include "OpenCLHelper/OpenCLHelper.h"
#include "Util/CommonHeaders.h"


namespace NN
{


//
//Base Layer, every layer is one of these
//


class BaseLayer
{
public:	
	static CLHelper::CLEnvironment* clEnvironment;
	
	std::string type;
	bool isMemoryAllocated;
	
	std::vector<BaseLayer*> nextL;
	std::vector<BaseLayer*> prevL;
	
	BaseLayer();
	bool ConnectOutput(BaseLayer* nxt);
	bool ConnectInput(BaseLayer* prv);
	BaseLayer* NextLayer(int inx=0);
	BaseLayer* PrevLayer(int inx=0);
	
	virtual void Allocate() = 0;
	virtual void ComputeForward() = 0;
	virtual void Backpropogate() = 0;
}; 



/*

//
//All functional layers are one to one eg. fully connected neuuron layer, conv layer, max pooling layer etc.
//

class OneToOneLayer: public BaseLayer
{
public:
	BaseLayer* prev;
	BaseLayer* next;

	OneToOneLayer();
	virtual void ComputeForward();
	virtual void Backpropogate();
};



//
//These layers are all glue. 
//one to many many to one layers need a stop and wait for the forwards/ backwards pass as there will be multiple paths. 
//

//one layer as input, output goes to many layers
class OneToManyLayer: public BaseLayer
{
public:
	BaseLayer* prev;
	std::vector<BaseLayer*> next;
	
	OneToOneLayer();
	virtual void ComputeForward();
	virtual void Backpropogate();
};

//many layers as input output is concat of inputs, goes to one layer
class ManyToOneLayer: public BaseLayer
{
public:
	std::vector<BaseLayer*> prev;
	BaseLayer* next;

	OneToOneLayer();
	virtual void ComputeForward();
	virtual void Backpropogate();
};


//output is a subset of inputs. This is actual a one to one layer
class MaskLayer: public OneToOneLayer
{
public:
	

	MaskLayer();
	virtual void ComputeForward();
	virtual void Backpropogate();
};

*/

}