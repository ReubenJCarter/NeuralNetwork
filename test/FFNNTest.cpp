#include "FFNN.h"
#include "fileAccess/matrixSaver.h"
#include "fileAccess/matrixLoader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void NandTest()
{
	printf("\n\n//*********NAND TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {1};
	ffnn.Create(2, 1, layerSize);
	printf("\nweights before:\n\n");
	ffnn.Print();
	//Train
	Matd trainingInput(2, 4);
	Matd trainingOutput(1, 4);
	
	trainingInput.Set(0, 0, 0);
	trainingInput.Set(1, 0, 0);
	trainingInput.Set(0, 1, 0);
	trainingInput.Set(1, 1, 1);
	trainingInput.Set(0, 2, 1);
	trainingInput.Set(1, 2, 0);
	trainingInput.Set(0, 3, 1);
	trainingInput.Set(1, 3, 1);
	
	trainingOutput.Set(0, 0, 1);
	trainingOutput.Set(0, 1, 0);
	trainingOutput.Set(0, 2, 0);
	trainingOutput.Set(0, 3, 0);
	
	for(int i = 0; i < 10000; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 1);
	
	//Test
	Matd testInput(2, 4); 
	Matd testOutput;
	
	testInput.Set(0, 0, 0);
	testInput.Set(1, 0, 0);
	testInput.Set(0, 1, 0);
	testInput.Set(1, 1, 1);
	testInput.Set(0, 2, 1);
	testInput.Set(1, 2, 0);
	testInput.Set(0, 3, 1);
	testInput.Set(1, 3, 1);
	
	testOutput = ffnn.ForwardUpdate(testInput);
	printf("\n\ninput to nn:\n");
	testInput.Print();
	printf("\n\n\noutput from neural network after training:\n");
	testOutput.Print();
	
	printf("\nweights after training:\n\n");
	ffnn.Print();
}

void XnorTest()
{
	printf("\n\n//*********XNOR TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {3, 1};
	ffnn.Create(2, 2, layerSize);
	printf("\nweights before:\n\n");
	ffnn.Print();
	//Train
	Matd trainingInput(2, 4);
	Matd trainingOutput(1, 4);
	
	trainingInput.Set(0, 0, 0);
	trainingInput.Set(1, 0, 0);
	trainingInput.Set(0, 1, 0);
	trainingInput.Set(1, 1, 1);
	trainingInput.Set(0, 2, 1);
	trainingInput.Set(1, 2, 0);
	trainingInput.Set(0, 3, 1);
	trainingInput.Set(1, 3, 1);
	
	trainingOutput.Set(0, 0, 1);
	trainingOutput.Set(0, 1, 0);
	trainingOutput.Set(0, 2, 0);
	trainingOutput.Set(0, 3, 1);
	
	for(int i = 0; i < 10000; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 1);
	
	//WriteBack
	ffnn.Save("testFile.csv");
	ffnn.Load("testFile.csv");
	
	//Test
	Matd testInput(2, 4); 
	Matd testOutput;
	
	testInput.Set(0, 0, 0);
	testInput.Set(1, 0, 0);
	testInput.Set(0, 1, 0);
	testInput.Set(1, 1, 1);
	testInput.Set(0, 2, 1);
	testInput.Set(1, 2, 0);
	testInput.Set(0, 3, 1);
	testInput.Set(1, 3, 1);
	
	testOutput = ffnn.ForwardUpdate(testInput);
	printf("\n\ninput to nn:\n");
	testInput.Print();
	printf("\n\n\noutput from neural network after training:\n");
	testOutput.Print();
	
	printf("\nweights after training:\n\n");
	ffnn.Print();	
}

double GetRandInRange(double fMin, double fMax)
{
	return( (fMax - fMin) * ((double)rand() / (double)RAND_MAX)) + fMin;
}

void SinTest()
{
	printf("\n\n//*********Sin TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {20, 1};
	ffnn.Create(1, 2, layerSize);
	ffnn.SetActivationFunction(1, FFNN::Linear, FFNN::GradientLinear);
	//Train
	
	Matd trainingInput(1, 100);
	Matd trainingOutput(1, 100);
	
	for(int i = 0 ; i < 100; i++)
	{
		double randNum = GetRandInRange(-3.141 * 2, 3.141 * 2);
		trainingInput.Set(0, i, randNum);
		trainingOutput.Set(0, i, sin(randNum));
	}
	
	int epochs = 2000;
	Matd cost(epochs, 2);
	for(int i = 0; i < epochs; i++)
	{
		ffnn.BackPropogate(trainingInput, trainingOutput, 0.25 * 100);
		cost.Set(i, 0, ffnn.GetCost(trainingInput, trainingOutput));
		cost.Set(i, 1, (double)i);
	}
	MatrixSaver saverCost("costFunction.csv");
	saverCost.WriteMatrix(cost);
	
	//Test
	Matd testOutput;
	Matd testInput(1, 100);
	for(int i = 0 ; i < 100; i++)
	{
		double randNum = GetRandInRange(-3.141 * 2, 3.141 * 2);
		testInput.Set(0, i, randNum);
	}
	testOutput = ffnn.ForwardUpdate(testInput);
	printf("\n\n\nideal output:\n");
	trainingOutput.Print();
	printf("\n\n\noutput from neural network after training:\n");
	testOutput.Print();
	
	Matd saveMatrix(4, 100);
	saveMatrix.Copy(testInput, 0, 0, 0, 0, 1, 100);
	saveMatrix.Copy(testOutput, 0, 0, 1, 0, 1, 100);
	saveMatrix.Copy(trainingInput, 0, 0, 2, 0, 1, 100);
	saveMatrix.Copy(trainingOutput, 0, 0, 3, 0, 1, 100);
	saveMatrix = Trans(saveMatrix);
	MatrixSaver saver("testFile.csv");
	saver.WriteMatrix(saveMatrix);
	
	printf("\nweights after training:\n\n");
	ffnn.Print();	
}

void SingleNeuronCostFunctionTest()
{
	printf("\n\n//*********COST FUNCTION TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {1};
	ffnn.Create(1, 1, layerSize);
	ffnn.SetWeight(0, 0, 0, 2.0);
	ffnn.SetWeight(0, 0, 1, 2.0);
	Matd trainIn(1, 1); 
	trainIn.Set(0, 0, 1.0);
	Matd trainOut(1, 1);
	trainOut.Set(0, 0, 0.0);
	int epochs = 300;
	Matd cost(epochs, 2);
	for(int i = 0; i < epochs; i++)
	{
		ffnn.BackPropogate(trainIn, trainOut, 0.15);
		cost.Set(i, 1, ffnn.GetCost(trainIn, trainOut));
		cost.Set(i, 0, (double)i);
	}
	MatrixSaver saverCost("costFunction2.csv");
	saverCost.WriteMatrix(cost);
}

int main()
{
	NandTest();
	XnorTest();
	SinTest();
	SingleNeuronCostFunctionTest();
	
	return 0;
}