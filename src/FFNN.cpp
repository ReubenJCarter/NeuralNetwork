#include "FFNN.h"
#include <stdio.h>
#include <math.h>

double FFNN::Square(double fX)
{
	return fX * fX;
}

double FFNN::FastExp(double fX) 
{
  //fX = 1.0 + fX / 1024;
  //fX *= fX; fX *= fX; fX *= fX; fX *= fX;
  //fX *= fX; fX *= fX; fX *= fX; fX *= fX;
  //fX *= fX; fX *= fX;
  return exp(fX);
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

double FFNN::Linear(double fX)
{
	return fX * 0.05;
}

double FFNN::GradientLinear(double fX)
{
	return 0.05;
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
	
	activationFunction = new CompFunction[layerNumber];
	gradientActivationFunction = new CompFunction[layerNumber];
	
	for(int i = 0; i < layerNumber; i++)
	{
		layer[i].Create(fLayerSizes[i] + 1, prevLayerSize);
		layer[i].Randomize(0.0, 1.0);
		prevLayerSize = fLayerSizes[i] + 1;
		layer[i].SetRow(layer[i].rows - 1, 0.0);
		layer[i].Set(layer[i].rows - 1, layer[i].cols - 1, 1.0);
		
		activationFunction[i] = Sigmiod;
		gradientActivationFunction[i] = GradientSigmiod;
	}
}

void FFNN::SetActivationFunction(int fLayer, CompFunction fActivationFunction, CompFunction fGradientActivationFunction)
{
	activationFunction[fLayer] = fActivationFunction; 
	gradientActivationFunction[fLayer] = fGradientActivationFunction; 
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
		output.ComponentFunction(activationFunction[i]);
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
	#define PRINT_OUTPUT 0
	
	#if PRINT_OUTPUT == 1
		printf("START BP\n\n");
	#endif
	
	Matd* z;
	Matd* a;
	Matd* delta;
	
	z = new Matd[layerNumber];
	a = new Matd[layerNumber];
	delta = new Matd[layerNumber]; 
	
	//Feed forward, computing all the values of z and a
	Matd aRun(fInput.rows + 1, fInput.cols);
	aRun.Copy(fInput, 0, 0);
	aRun.SetRow(aRun.rows - 1, 1.0);
	
	#if PRINT_OUTPUT == 1
		printf("\nfirst a:\n");
		aRun.Print();
	#endif
	
	for(int i = 0; i < layerNumber; i++)
	{
		aRun = layer[i] * aRun;
		z[i] = aRun;
		aRun.ComponentFunction(activationFunction[i]);
		aRun.SetRow(aRun.rows - 1, 1.0);
		a[i] = aRun;
		
		#if PRINT_OUTPUT == 1
			printf("\nweights layer %d:\n", i);
			layer[i].Print();
			printf("\nz layer %d:\n", i);
			z[i].Print();
			printf("\na layer %d:\n", i);
			a[i].Print();
		#endif
	}
	
	//Compute output error
	Matd y(fTrainingOutput.rows + 1, fTrainingOutput.cols);
	y.Copy(fTrainingOutput, 0, 0);
	y.SetRow(y.rows - 1, 1.0);
	
	int L = layerNumber - 1;
	z[L].ComponentFunction(gradientActivationFunction[L]);
	Matd outputDifference;
	outputDifference = a[L] - y;
	delta[L] = HadProd(outputDifference, z[L]);

	//Gradient decent to adjust weights
	Matd trIn(fInput.rows + 1, fInput.cols);
	trIn.Copy(fInput, 0, 0);//copy input 
	trIn.SetRow(trIn.rows - 1, 1.0);
	Matd product;
	if(L > 0)
		product = delta[L] * Trans(a[L - 1]);
	else
		product = delta[L] * Trans(trIn);
	layer[L] -= product * (fLearningRate / fInput.cols);
		
	#if PRINT_OUTPUT == 1
		printf("\ntraining output:\n");
		y.Print();
		printf("\noutput difference:\n");
		outputDifference.Print();
		printf("\nderivative activation function\n");
		z[L].Print();
		printf("\nneuron errors layer %d:\n", L);
		delta[L].Print();		
		printf("\na[%d]:\n", L - 1);
		if(L > 0) a[L - 1].Print(); else trIn.Print();
		printf("\ndelta * trans(a[%d]:\n", L - 1);
		product.Print();
		printf("\ngradient decented layer %d:\n", L);
		layer[L].Print();
	#endif
	
	for(int i = L - 1; i >= 0; i--)
	{
		//Back Propagate the error
		z[i].ComponentFunction(gradientActivationFunction[i]);
		z[i].SetRow(z[i].rows - 1, 0.0);
		Matd transMult;
		transMult = Trans(layer[i + 1]) * delta[i + 1];
		delta[i] = HadProd(transMult, z[i]);
		//Gradient decent to adjust weights
		Matd product;
		if(i > 0)
			product = delta[i] * Trans(a[i - 1]);
		else
			product = delta[i] * Trans(trIn);
		layer[i] -= product * (fLearningRate / fInput.cols);
			
		#if PRINT_OUTPUT == 1
			printf("\nderivative z[%d]\n", i);
			z[i].Print();
			printf("\ndelta[%d]\n", i);
			delta[i].Print();
			printf("\ngradient decented layer %d:\n", L);
			layer[i].Print();
		#endif
	}
	
	#if PRINT_OUTPUT == 1
		printf("END BP\n");
	#endif
	delete[] z;
	delete[] a;
	delete[] delta;
}

void FFNN::Save(const char* fFN)
{
	FILE* fptr;
	fptr = fopen(fFN, "wb");
	if(fptr != NULL)
	{
		fprintf(fptr, "%d\n", layerNumber);//layerNumber
		for(int i = 0; i < layerNumber; i++)
		{
			fprintf(fptr, "%d,%d\n", layer[i].rows, layer[i].cols);
			for(int k = 0; k < layer[i].rows; k++)
			{
				for(int j = 0; j < layer[i].cols; j++)
				{
					fprintf(fptr, "%f", layer[i].Get(k, j));
					if(j < layer[i].cols - 1)
						fprintf(fptr, ",");
					else
						fprintf(fptr, "\n");
				}
			}
		}
		fclose(fptr);
	}
}

void FFNN::Load(const char* fFN)
{
	Destroy();
	FILE* fptr;
	fptr = fopen(fFN, "rb");
	if(fptr != NULL)
	{
		fscanf(fptr, "%d\n", &layerNumber);
		layer = new Matd[layerNumber];
		
		for(int i = 0; i < layerNumber; i++)
		{
			int r, c;
			fscanf(fptr, "%d,%d\n", &r, &c);
			layer[i].Create(r, c);
			for(int k = 0; k < layer[i].rows; k++)
			{
				for(int j = 0; j < layer[i].cols; j++)
				{
					double val;
					if(j < layer[i].cols - 1)
						fscanf(fptr, "%lf,", &val);
					else
						fscanf(fptr, "%lf\n", &val);
					layer[i].Set(k, j, val);
				}
			}
		}
		fclose(fptr);
	}
}