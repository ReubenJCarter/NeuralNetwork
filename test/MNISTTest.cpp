#include "Util/CommonHeaders.h"
#include "Util/MNISTImageFile.h"
#include "Util/MNISTLabelFile.h"
#include "FullyConnectedLayer.h"
#include "InputLayer.h"
	
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>


using namespace NN;


int main(int argc, char* argv[])
{
	//
	//Init Cl
	//
	CLHelper::CLEnvironment clEnv;
	BaseLayer::clEnvironment = &clEnv;
	FullyConnectedLayer::Init();
	
	//
	//CMD check
	//
	if(argc != 5)
	{
		std::cerr << "arguments should be MNISTTest <trainImageFile> <trainLabelFile> <testImageFile> <testLabelFile>" << std::endl;
		return 0; 
	}
	
	//
	//open MNIST training and testing sets
	//
	Util::MNISTImageFile trainFile;
	bool trainFileOpen = trainFile.Open(argv[1]);
	Util::MNISTLabelFile trainLabelFile;
	bool trainLabelFileOpen = trainLabelFile.Open(argv[2]);
	Util::MNISTImageFile testFile;
	bool testFileOpen = testFile.Open(argv[3]);
	Util::MNISTLabelFile testLabelFile;
	bool testLabelFileOpen = testLabelFile.Open(argv[4]);
	if(!trainFileOpen || !trainFileOpen || !testFileOpen || !testLabelFileOpen)
	{
		std::cerr << "image file or label file not open" << std::endl;
		return 1;
	}
	int trainingImageNumber = trainFile.GetImageNumber();
	int testImageNumber = testFile.GetImageNumber();
	if(trainingImageNumber != trainLabelFile.GetLabelNumber() || testImageNumber != testLabelFile.GetLabelNumber())
	{
		std::cerr << "Image number is not the same as label number" << std::endl;
		return 1;
	}
	int imageWidth = trainFile.GetWidth();
	int imageHeight = trainFile.GetHeight();

	//
	//create network
	//
	int miniBatchSize = 10;
	
	std::cout << "Creating input layer " << (imageWidth * imageHeight) << " X " << miniBatchSize << std::endl;
	InputLayer inputLayer;
	inputLayer.SetSize(imageWidth * imageHeight, miniBatchSize);
	
	//std::cout << "Setting input values" << std::endl;
	//inputLayer.SetInput(&inputLayerBuffer[0]);
	
	std::cout << "Creating fully connected layer" << std::endl;
	FullyConnectedLayer fcLayer0; 
	fcLayer0.activationType = FullyConnectedLayer::LOGISTIC;
	fcLayer0.SetSize(15);
	FullyConnectedLayer fcLayer1; 
	fcLayer1.activationType = FullyConnectedLayer::LOGISTIC;
	fcLayer1.SetSize(10);
	
	std::cout << "Connecting up layers" << std::endl;
	fcLayer0.ConnectInput(&inputLayer);
	fcLayer1.ConnectInput(&fcLayer0);
	
	std::cout << "Allocating layers" << std::endl;
	inputLayer.Allocate();
	std::cout << "Allocating layers" << std::endl;
	fcLayer0.Allocate();
	std::cout << "Allocating layers" << std::endl;
	fcLayer1.Allocate();
	
	std::cout << "Setting fully connected layer weights and biases" << std::endl;
	fcLayer0.RandomizeWeights(0, 1, 0, 1);
	fcLayer1.RandomizeWeights(0, 1, 0, 1);
	
	//
	//train the network
	//
	unsigned int trainingNumber = 50000;
	unsigned int epochs = 50;
	
	std::vector<float> testIn(imageWidth * imageHeight * miniBatchSize, 1);
	std::vector<float> testOut(10 * miniBatchSize);
		
	fcLayer0.learningRate = 0.5; 
	fcLayer1.learningRate = 0.5; 
		
	std::vector<float> errors(imageWidth * imageHeight * miniBatchSize, 1);
		
	for(int e = 0; e < epochs; e++)
	{
		std::cout << "epoch " << e << std::endl;
		//build index array of random order indexes to the training data
		std::vector<int> trainingSetIndecies(trainingNumber);
		std::iota(trainingSetIndecies.begin(), trainingSetIndecies.end(), 0);
		std::random_shuffle(trainingSetIndecies.begin(), trainingSetIndecies.end());
		
		for(int m = 0; m < trainingNumber / miniBatchSize; m++)
		{
			//take next 10 indicies and run as minibatch 
			//std::cout << "Loading Train Data" << std::endl;
			std::vector<float> miniBatchIn(imageWidth * imageHeight * miniBatchSize);
			std::vector<float> miniBatchOut(10 * miniBatchSize, 0);
			for(int i = 0; i < miniBatchSize; i++)
			{
				//get training img 
				trainFile.GetImageDataAsFloat(trainingSetIndecies[m * miniBatchSize + i], &miniBatchIn[imageWidth * imageHeight * i]);
			
				//set output from lable
				uint8_t label;
				trainLabelFile.GetLabelData(trainingSetIndecies[m * miniBatchSize + i], label);
				miniBatchOut[10 * i + (int)label] = 1;
			}
			
			//
			//Train the Net
			//
			
			//forward propogate
			//std::cout << "Forward Propogate" << std::endl;
			inputLayer.SetInput(&miniBatchIn[0]);
			inputLayer.ComputeForward();
			fcLayer0.ComputeForward();
			fcLayer1.ComputeForward();
			
			//back propogate error
			//std::cout << "Backwards Propogate" << std::endl;
			fcLayer1.ComputeOutputError(&miniBatchOut[0]); 
			//std::cout << "Backwards Propogate" << std::endl;
			fcLayer0.Backpropogate();
			
			//update weights and biases
			//std::cout << "Weights Biases Update" << std::endl;
			fcLayer0.AdjustWeightsBiases();
			//std::cout << "Weights Biases Update" << std::endl;
			fcLayer1.AdjustWeightsBiases();
			
		}
		
		//Test
		std::cout << "Running Tests" << std::endl;
		unsigned int testNumber = 10000;
		unsigned int correctNumber = 0;
		
		for(unsigned int t = 0; t < testNumber; t++)
		{
			
			//std::cout << "Load Test Image Data" << std::endl;
			testFile.GetImageDataAsFloat(t, &testIn[0]);
			
			//forward propogate
			//std::cout << "Forward Propogate" << std::endl;
			inputLayer.SetInput(&testIn[0]);
			inputLayer.ComputeForward();
			fcLayer0.ComputeForward();
			fcLayer1.ComputeForward();
			fcLayer1.ReadOutput(&testOut[0]);
			
			int output = -1;
			for(int i = 0; i < 10; i++)
			{
				if(testOut[i] > 0.5) output = i;
			}
			uint8_t label;
			testLabelFile.GetLabelData(t, label);
			if((int)label == output) correctNumber++;
		}
		std::cout << "Correct out of " << testNumber << "=" << correctNumber << std::endl;
		
	}
	
	return 1;
}