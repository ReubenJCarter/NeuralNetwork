#include "FFNN.h"
#include <stdio.h>

double FFNN::FastExp(double fX) 
{
  fX = 1.0 + fX / 1024;
  fX *= fX; fX *= fX; fX *= fX; fX *= fX;
  fX *= fX; fX *= fX; fX *= fX; fX *= fX;
  fX *= fX; fX *= fX;
  return fX;
}

double FFNN::Sigmiod(double fX)
{
	return 1.0 / (1.0 + FastExp(-fX));
}

double FFNN::GradientSigmiod(double fX)
{
	double sigm = 1.0 / (1.0 + FastExp(-fX));
	return (1 - sigm) * sigm;
}

FFNN::FFNN()
{
	layerNumber = 0;
}

FFNN::~FFNN()
{
	Destroy();
}

void FFNN::Create(int fInputNumber, int fLayerNumber, const int fLayerSizes[])
{
	Destroy();
	layerNumber = fLayerNumber;
	layer = new Matd[layerNumber];
	int prevLayerSize = fInputNumber + 1;
	for(int i = 0; i < layerNumber; i++)
	{
		layer[i].Create(fLayerSizes[i] + 1, prevLayerSize);
		layer[i].Randomize(0.0, 1.0);
		prevLayerSize = fLayerSizes[i] + 1;
		layer[i].SetRow(layer[i].rows - 1, 0.0);
		layer[i].Set(layer[i].rows - 1, layer[i].cols - 1, 1.0);
	}
}

void FFNN::Destroy()
{
	if(layerNumber != 0)
	{
		layerNumber = 0;
		delete[] layer;
	}
}

Matd FFNN::ForwardUpdate(Matd& fInput)
{
	Matd output(fInput.rows + 1, fInput.cols);
	output.SetRow(fInput.rows, 1.0);
	output.Copy(fInput, 0, 0);
	for(int i = 0; i < layerNumber; i++)
	{
		output = layer[i] * output;
		output.ComponentFunction(Sigmiod);
		output.SetRow(output.rows - 1, 1.0);
	}
	return output;
}

void FFNN::Print()
{
	for(int i = 0; i < layerNumber; i++)
	{
		printf("layer %d:\n", i);
		layer[i].Print();
	}
}

void FFNN::BackPropogate(Matd& fInput, Matd& fTrainingOutput, double fLearningRate)
{
	printf("START BP\n\n");
	Matd* z;
	Matd* a;
	Matd* delta;
	Matd y(fTrainingOutput.rows + 1, fTrainingOutput.cols);
	
	z = new Matd[layerNumber];
	a = new Matd[layerNumber];
	delta = new Matd[layerNumber]; 
	y.Copy(fTrainingOutput, 0, 0);
	y.SetRow(y.rows - 1, 1.0);
	
	//Feed forward, computing all the values of z and a
	Matd aRun(fInput.rows + 1, fInput.cols);
	aRun.Copy(fInput, 0, 0);
	aRun.SetRow(aRun.rows - 1, 1.0);
	
	printf("first a:\n");
	aRun.Print();
	
	for(int i = 0; i < layerNumber; i++)
	{
		aRun = layer[i] * aRun;
		z[i] = aRun;
		printf("z layer %d:\n", i);
		z[i].Print();
		printf("weights layer %d:\n", i);
		layer[i].Print();
		aRun.ComponentFunction(Sigmiod);
		aRun.SetRow(aRun.rows - 1, 1.0);
		a[i] = aRun;
		printf("a layer %d:\n", i);
		a[i].Print();
	}
	
	//Compute output error
	int L = layerNumber - 1;
	z[L].ComponentFunction(GradientSigmiod);
	z[L].SetRow(z[L].rows - 1, 1.0);
	delta[L] = HadProd(a[L] - y, z[L]);
	//Gradient decent to adjust weights
	layer[L] -= (delta[L] * Trans(a[L])) * (fLearningRate / fInput.cols);
	//Back Propagate the error
	for(int i = L - 1; i >= 0; i--)
	{
		z[i].ComponentFunction(GradientSigmiod);
		z[i].SetRow(z[i].rows - 1, 1.0);
		delta[i] = HadProd(MultTransA(layer[i + 1], delta[i + 1]), z[i]);
		//Gradient decent to adjust weights
		layer[i] -= (delta[i] * Trans(a[i])) * (fLearningRate / fInput.cols);
	}
	
	printf("END BP\n");
}