#include "Util/CommonHeaders.h"
#include "FullyConnectedLayer.h"
#include "InputLayer.h"


using namespace NN;


int main(int argc, char* argv[])
{
	CLHelper::CLEnvironment clEnv;
	BaseLayer::clEnvironment = &clEnv;
	FullyConnectedLayer::Init();
	
	std::cout << "Creating input layer" << std::endl;
	InputLayer inputLayer;
	inputLayer.SetSize(10, 1);
	
	std::cout << "Setting input values" << std::endl;
	inputLayer.input.resize(10);
	for(int i = 0; i < 10; i++)
	{
		inputLayer.input[i] = (float)i; 
	}
	
	std::cout << "Creating fully connected layer" << std::endl;
	FullyConnectedLayer fcLayer; 
	fcLayer.SetSize(1);
	
	std::cout << "Connecting up layers" << std::endl;
	fcLayer.ConnectInput(&inputLayer);
	
	std::cout << "Allocating layer" << std::endl;
	inputLayer.Allocate();
	fcLayer.Allocate();
	
	std::cout << "Compute Forward" << std::endl;
	inputLayer.ComputeForward();
	fcLayer.ComputeForward();
	
	return 1;
}