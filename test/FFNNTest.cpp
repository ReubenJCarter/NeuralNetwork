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
	trainingInput.Set(0, 1, 1);
	trainingOutput.Set(0, 0, 1);
	trainingOutput.Set(0, 1, 0);
	
	ffnn.BackPropogate(trainingInput, trainingOutput, 1);
	
	//Test
	Matd testInput(1, 1); 
	Matd testOutput;
	
	testInput.Set(0, 0, 0);
	testOutput = ffnn.ForwardUpdate(testInput);
	//testOutput.Print();
	testInput.Set(0, 0, 1);
	testOutput = ffnn.ForwardUpdate(testInput);
	//testOutput.Print();
	
	//ffnn.Print();
	
	return 0;
}