#include "FFNN.h"
#include <stdio.h>

int main()
{
	FFNN ffnn;
	const int layerSize[] = {1};
	ffnn.Create(1, 1, layerSize);
	//ffnn.Print();
	
	//Train
	Matd trainingInput(1, 2);
	Matd trainingOutput(1, 2);
	trainingInput.Set(0, 0, 0);
	trainingInput.Set(0, 1, 10);
	trainingOutput.Set(0, 0, 1);
	trainingOutput.Set(0, 1, 0);
	
	for(int i = 0; i < 1000; i++)
		ffnn.BackPropogate(trainingInput, trainingOutput, 10);
	
	//Test
	Matd testInput(1, 2); 
	Matd testOutput;
	
	testInput.Set(0, 0, 0);
	testInput.Set(0, 1, 1);
	testOutput = ffnn.ForwardUpdate(testInput);
	printf("\n\ninput to nn:\n");
	testInput.Print();
	printf("\n\n\noutput from neural network after training:\n");
	testOutput.Print();
	
	//ffnn.Print();
	
	return 0;
}