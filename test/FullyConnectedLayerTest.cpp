#include "FullyConnectedLayer.h"


using namespace NN;


int main(int argc, char* argv[])
{
	CLHelper::CLEnvironment clEnv;
	BaseLayer::clEnvironment = &clEnv;
	FullyConnectedLayer::Init();
	
	FullyConnectedLayer testLayer; 
	testLayer.layerSize = 10;
	testLayer.layerThickness = 1;
	
	
	
	return 1;
}