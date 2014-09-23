#include "FFNN.h"
#include <stdio.h>

void AndTest()
{
	printf("\n\n//*********AND TEST**********//\n");
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

int main()
{
	AndTest();
	
	FFNN ffnn;
	const int layerSize[] = {1};
	ffnn.Create(2, 1, layerSize);
	
	printf("\nweights before:\n\n");
	ffnn.Print();
	
	//Train
	Matd trainingInput(2, 2);
	Matd trainingOutput(1, 2);
	
	trainingInput.Set(0, 0, 0);
	trainingInput.Set(1, 0, 0);
	trainingInput.Set(0, 1, 1);
	trainingInput.Set(1, 1, 1);
	
	trainingOutput.Set(0, 0, 1);
	trainingOutput.Set(0, 1, 0);
	
	for(int i = 0; i < 1; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 1);
	
	//Test
	Matd testInput(2, 2); 
	Matd testOutput;
	
	testInput.Set(0, 0, 0);
	testInput.Set(1, 0, 0);
	testInput.Set(0, 1, 1);
	testInput.Set(1, 1, 1);
	
	testOutput = ffnn.ForwardUpdate(testInput);
	printf("\n\ninput to nn:\n");
	testInput.Print();
	printf("\n\n\noutput from neural network after training:\n");
	testOutput.Print();
	
	printf("\nweights after:\n\n");
	ffnn.Print();
	
	return 0;
}