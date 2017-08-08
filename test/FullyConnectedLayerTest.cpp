#include "FullyConnectedLayer.h"


using namespace NN;


int main(int argc, char* argv[])
{
	CLHelper::CLEnvironment clEnv;
	BaseLayer::clEnvironment = &clEnv;
	FullyConnectedLayer::Init();
	
	FullyConnectedLayer testLayer; 
	
	
	
	return 1;
}