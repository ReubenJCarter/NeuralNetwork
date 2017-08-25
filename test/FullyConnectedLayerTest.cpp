#include "Util/CommonHeaders.h"
#include "FullyConnectedLayer.h"
#include "InputLayer.h"
	
	
using namespace NN;
	
	
void Test0()
{
	std::cout << "Creating input layer" << std::endl;
	InputLayer inputLayer;
	inputLayer.SetSize(10, 1);
	
	std::cout << "Setting input values" << std::endl;
	std::vector<float> inputLayerBuffer(10);
	for(int i = 0; i < 10; i++)
	{
		inputLayerBuffer[i] = (float)i; 
	}
	inputLayer.SetInput(&inputLayerBuffer[0]);
	
	std::cout << "Creating fully connected layer" << std::endl;
	FullyConnectedLayer fcLayer; 
	fcLayer.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer.SetSize(2);
	
	std::cout << "Connecting up layers" << std::endl;
	fcLayer.ConnectInput(&inputLayer);
	
	std::cout << "Allocating layer" << std::endl;
	inputLayer.Allocate();
	fcLayer.Allocate();
	
	std::cout << "Setting fully connected layer weights and biases" << std::endl;
	fcLayer.RandomizeWeights(1, 0.0000001, 100, 0.0000001);
	
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
	
	std::cout << "if at this point the output is 145 for both all is well" << std::endl;
}


void Test1()
{
	//Create Input Layer
	std::cout << "Creating input layer" << std::endl;
	InputLayer inputLayer;
	inputLayer.SetSize(4, 2);
	
	std::cout << "Setting input values" << std::endl;
	std::vector<float> inputLayerBuffer(4 * 2);
	for(int i = 0; i < 4 * 2; i++)
	{
		inputLayerBuffer[i] = (float)i; 
	}
	inputLayer.SetInput(&inputLayerBuffer[0]);

	
	//Create FC layers
	std::cout << "Creating fully connected layer 0" << std::endl;
	FullyConnectedLayer fcLayer0; 
	fcLayer0.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer0.SetSize(1);
	
	std::cout << "Creating fully connected layer 1" << std::endl;
	FullyConnectedLayer fcLayer1; 
	fcLayer1.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer1.SetSize(1);
	
	std::cout << "Creating fully connected layer 2" << std::endl;
	FullyConnectedLayer fcLayer2; 
	fcLayer2.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer2.SetSize(1);
	
	//Connect Layers
	std::cout << "Connecting up layers" << std::endl;
	inputLayer.ConnectOutput(&fcLayer0);
	fcLayer0.ConnectOutput(&fcLayer1);
	fcLayer1.ConnectOutput(&fcLayer2);
	
	//Allocate Layers
	std::cout << "Allocating layer" << std::endl;
	inputLayer.Allocate();
	fcLayer0.Allocate();
	fcLayer1.Allocate();
	fcLayer2.Allocate();
	
	//Set up weights and biases
	std::cout << "Setting fully connected layers weights and biases" << std::endl;
	fcLayer0.RandomizeWeights(1, 0.0001, 0, 0.00001);
	fcLayer1.RandomizeWeights(1, 0.0001, 0, 0.00001);
	fcLayer2.RandomizeWeights(1, 0.0001, 0, 0.00001);
	
	//Forward pass
	std::cout << "Compute Forward" << std::endl;
	inputLayer.ComputeForward();
	fcLayer0.ComputeForward();
	fcLayer1.ComputeForward();
	fcLayer2.ComputeForward();
	
	//Read back last layer
	std::cout << "Read Final FC Layer:" << std::endl;
	std::vector<float> outputBuffer(fcLayer2.layerSize * fcLayer2.layerThickness);
	fcLayer2.ReadOutput(&outputBuffer[0]);
	for(int i = 0; i < fcLayer2.layerSize * fcLayer2.layerThickness; i++)
	{
		std::cout << outputBuffer[i] << std::endl;
	}
	
	std::cout << "if at this point the output is 6 , 22 all is well" << std::endl;
}

void Test2()
{
	//Create Input Layer
	std::cout << "Creating Large input layer" << std::endl;
	InputLayer inputLayer;
	inputLayer.SetSize(10000, 10);
	
	std::cout << "Setting input values" << std::endl;
	std::vector<float> inputLayerBuffer(10000 * 10);
	for(int i = 0; i < 10000 * 10; i++)
	{
		inputLayerBuffer[i] = (float)i; 
	}
	inputLayer.SetInput(&inputLayerBuffer[0]);

	
	//Create FC layers
	std::cout << "Creating fully connected layer 0" << std::endl;
	FullyConnectedLayer fcLayer0; 
	fcLayer0.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer0.SetSize(1000);
	
	std::cout << "Creating fully connected layer 1" << std::endl;
	FullyConnectedLayer fcLayer1; 
	fcLayer1.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer1.SetSize(500);
	
	std::cout << "Creating fully connected layer 2" << std::endl;
	FullyConnectedLayer fcLayer2; 
	fcLayer2.activationType = FullyConnectedLayer::IDENTITY;
	fcLayer2.SetSize(1);
	
	//Connect Layers
	std::cout << "Connecting up layers" << std::endl;
	inputLayer.ConnectOutput(&fcLayer0);
	fcLayer0.ConnectOutput(&fcLayer1);
	fcLayer1.ConnectOutput(&fcLayer2);
	
	//Allocate Layers
	std::cout << "Allocating layer" << std::endl;
	inputLayer.Allocate();
	fcLayer0.Allocate();
	fcLayer1.Allocate();
	fcLayer2.Allocate();
	
	//Set up weights and biases
	std::cout << "Setting fully connected layers weights and biases" << std::endl;
	fcLayer0.RandomizeWeights(1, 0.0001, 0, 0.0001);
	fcLayer1.RandomizeWeights(1, 0.0001, 0, 0.0001);
	fcLayer2.RandomizeWeights(1, 0.0001, 0, 0.0001);
	
	//Forward pass
	std::cout << "Compute Forward" << std::endl;
	inputLayer.ComputeForward();
	fcLayer0.ComputeForward();
	fcLayer1.ComputeForward();
	fcLayer2.ComputeForward();
	
	//Read back last layer
	std::cout << "Read Final FC Layer:" << std::endl;
	std::vector<float> outputBuffer(fcLayer2.layerSize * fcLayer2.layerThickness);
	fcLayer2.ReadOutput(&outputBuffer[0]);
	for(int i = 0; i < fcLayer2.layerSize * fcLayer2.layerThickness; i++)
	{
		std::cout << outputBuffer[i] << std::endl;
	}
}
	
int main(int argc, char* argv[])
{	
	//Setup Env
	CLHelper::CLEnvironment clEnv;
	BaseLayer::clEnvironment = &clEnv;
	FullyConnectedLayer::Init();
	
	//
	//Test 0
	//
	Test0(); 
	
	
	//
	//Test 1
	//
	Test1();
	
	
	//
	//Test 2
	//
	Test2();
	
	
	return 1;
}