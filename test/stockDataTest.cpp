#include "FFNN.h"
#include "fileAccess/matrixSaver.h"
#include "fileAccess/matrixLoader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main()
{
	printf("\n\n//*********Market data TEST**********//\n");
	FFNN ffnn;
	const int layerSize[] = {20, 1};
	ffnn.Create(1, 2, layerSize);
	ffnn.SetActivationFunction(1, FFNN::Linear, FFNN::GradientLinear);
	int trainSize = 100;
	int paramsIn = 4; 
	int paramsOut = 1; 
	Mat trainInput(trainSize, paramsIn);
	Mat trainOutput(trainSize, paramsOut);
	MatrixLoader load("../data/table.csv");
	load.ReadRowFromVertical(trainInput, 0, 1, 1, trainSize);//open
	load.ReadRowFromVertical(trainInput, 1, 1, 2, trainSize);//high
	load.ReadRowFromVertical(trainInput, 2, 1, 3, trainSize);//low
	load.ReadRowFromVertical(trainInput, 3, 1, 4, trainSize);//close
	load.ReadRowFromVertical(trainOutput, 0, 1, 5, trainSize);//volume
	return 0; 
}