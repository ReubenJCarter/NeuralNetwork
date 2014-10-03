#include "FFNN.h"
#include "matrixSaver.h"
#include "matrixLoader.h"
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
		double randNum = (4 * 3.141 * ((double)rand() / (double)RAND_MAX)) - 3.141 * 2;
		trainingInput.Set(0, i, randNum);
		trainingOutput.Set(0, i, sin(randNum));
	}
	
	for(int i = 0; i < 5000; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 0.25 * 100);
	
	//Test
	Matd testOutput;
	Matd testInput(1, 100);
	for(int i = 0 ; i < 100; i++)
	{
		double randNum = (4 * 3.141 * ((double)rand() / (double)RAND_MAX)) - 3.141 * 2;
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

int main()
{
	NandTest();
	XnorTest();
	SinTest();
	
	return 0;
}