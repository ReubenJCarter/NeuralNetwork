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
	
	std::cout << "Creating input layer" << std::endl;
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
	fcLayer0.Allocate();
	fcLayer1.Allocate();
	
	std::cout << "Setting fully connected layer weights and biases" << std::endl;
	fcLayer0.RandomizeWeights(1, 0, 100, 0);
	fcLayer1.RandomizeWeights(1, 0, 100, 0);
	
	return 1;
}