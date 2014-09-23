#include "FFNN.h"
#include <stdio.h>
#include <math.h>

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
	printf("\n\n//*********XNOR TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {6, 1};
	ffnn.Create(1, 2, layerSize);
	//Train
	
	Matd trainingInput(1, 360);
	Matd trainingOutput(1, 360);
	
	for(int i = 0 ; i < 360; i++)
	{
		trainingInput.Set(0, i, (double)i);
		trainingOutput.Set(0, i, (1.0 + sin(3.1415 * (double)i / 180.0)) / 2.0);
	}
	
	for(int i = 0; i < 1000; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 1);
	
	//Test
	Matd testOutput;
	
	testOutput = ffnn.ForwardUpdate(trainingInput);
	printf("\n\ninput to nn:\n");
	printf("\n\n\noutput from neural network after training:\n");
	trainingOutput.Print();
	testOutput.Print();
	//MatrixSaver saver("testFile.csv");
	//saver.WriteMatrix(mat);
	
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