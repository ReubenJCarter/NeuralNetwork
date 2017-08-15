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
	fcLayer.SetSize(2);
	
	std::cout << "Connecting up layers" << std::endl;
	fcLayer.ConnectInput(&inputLayer);
	
	std::cout << "Allocating layer" << std::endl;
	inputLayer.Allocate();
	fcLayer.Allocate();
	
	std::cout << "Randomising fully connected layer" << std::endl;
	fcLayer.RandomizeWeights(0, 1, 0, 1);
	
	std::cout << "Compute Forward" << std::endl;
	inputLayer.ComputeForward();
	fcLayer.ComputeForward();
	
	std::cout << "Read Input Layer:" << std::endl;
	std::vector<float> output0(inputLayer.layerSize);
	inputLayer.ReadOutput(&output0[0]);
	for(int i = 0; i < inputLayer.layerSize; i++)
	{
		std::cout << output0[i] << std::endl;
	}
	
	std::cout << "Read FC Layer:" << std::endl;
	std::vector<float> output1(fcLayer.layerSize);
	fcLayer.ReadOutput(&output1[0]);
	for(int i = 0; i < fcLayer.layerSize; i++)
	{
		std::cout << output1[i] << std::endl;
	}
	
	return 1;
}