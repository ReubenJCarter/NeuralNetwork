#define CBLAS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef CBLAS
#include <cblas.h>
#endif

class Matd
{
	public:
		int rows;
		int cols;
		int size;
		double* data;
		
		Matd();
		Matd(int fRow, int fCol);
		~Matd();
		double Get(int fRow, int fCol);
		void Set(int fRow, int fCol, double fVal);
		void Create(int fRow, int fCol);
		void Destroy();
		void operator=(const Matd& fMat);
		friend Matd operator*(const Matd& fA, const Matd& fB);
		friend Matd operator*(const Matd& fMat, const double fVal);
		friend Matd operator*(const double fVal, const Matd& fMat);
		friend Matd operator/(const Matd& fMat, const double fVal);
		friend Matd operator+(const Matd& fA, const Matd& fB);
		friend Matd operator-(const Matd& fA, const Matd& fB);
		void operator+=(const Matd& fMat);
		void operator-=(const Matd& fMat);
		friend Matd HadProd(const Matd& fA, const Matd& fB);
		friend Matd Trans(const Matd& fA);
		friend Matd MultTransA(const Matd& fA, const Matd& fB);
		friend Matd OuterProduct(const Matd& fA, const Matd& fB);
		void Print();
		void Randomize(double fMin, double fMax);
		void SetAll(double fVal);
		void SetRow(int fRow, double fVal);
		void SetCol(int fCol, double fVal);
		void ComponentFunction(double (*Func)(double a));
		void Copy(Matd& fMat, int fRowOffset, int fColOffset);
		void LoadCSV();
};

Matd::Matd()
{
	rows = 0; 
	cols = 0;
	size = 0;
}

Matd::Matd(int fRow, int fCol)
{
	size = 0;
	Create(fRow, fCol);
}

Matd::~Matd()
{
	Destroy();
}

double Matd::Get(int fRow, int fCol)
{
	int inx = rows * fCol + fRow;
	return data[inx];
}

void Matd::Set(int fRow, int fCol, double fVal)
{
	int inx = rows * fCol + fRow;
	data[inx] = fVal;
}

void Matd::Create(int fRow, int fCol)
{
	Destroy();
	rows = fRow;
	cols = fCol;
	size = rows * cols;
	if(size != 0)
	{
		data = new double[size];
	}
}

void Matd::Destroy()
{
	if(size != 0)
	{
		rows = 0; 
		cols = 0;
		size = 0;
		delete[] data;
	}
}

void Matd::operator=(const Matd& fMat)
{
	if(size != fMat.size)
	{
		Create(fMat.rows, fMat.cols);
	}
	else
	{
		rows = fMat.rows;
		cols = fMat.cols;
	}
	int colOffset;
	for(int i = 0; i < cols; i++)
	{
		colOffset = i * rows;
		for(int j = 0; j < rows; j++)
		{
			data[colOffset + j] = fMat.data[colOffset + j];
		}
	}
}

Matd operator*(const Matd& fA, const Matd& fB)
{
	//A B
	int r = fA.rows;
	int c = fB.cols;
	Matd out(r, c);
	
#ifndef CBLAS
	int colOffOut; 
	int colOffsetB;
	double sum;
	for(int i = 0; i < c; i++)
	{
		colOffOut = i * r;
		colOffsetB = i * fB.rows;
		for(int j = 0; j < r; j++)
		{
			sum = 0; 
			for(int k = 0; k < fB.rows; k++)
			{
				sum += fA.data[k * r + j] * fB.data[colOffsetB + k];
			}
			out.data[colOffOut + j] = sum;
		}
	}
#else
	CBLAS_ORDER ord = CblasColMajor;
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	cblas_dgemm(ord, trans, trans, fA.rows, fB.cols, fA.cols, 1.0, fA.data, fA.rows, fB.data, fB.rows, 0, out.data, r);
#endif

	return out;
}

Matd operator*(const Matd& fMat, const double fVal)
{
	Matd out(fMat.rows, fMat.cols);
	int collOff;
	for(int i = 0; i < fMat.cols; i++)
	{
		collOff = i * fMat.rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			out.data[collOff + j] = fMat.data[collOff + j] * fVal;
		}
	}
	return out;
}

Matd operator*(const double fVal, const Matd& fMat)
{
	Matd out(fMat.rows, fMat.cols);
	int collOff;
	for(int i = 0; i < fMat.cols; i++)
	{
		collOff = i * fMat.rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			out.data[collOff + j] = fMat.data[collOff + j] * fVal;
		}
	}
	return out;
}

Matd operator/(const Matd& fMat, const double fVal)
{
	Matd out(fMat.rows, fMat.cols);
	int collOff;
	for(int i = 0; i < fMat.cols; i++)
	{
		collOff = i * fMat.rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			out.data[collOff + j] = fMat.data[collOff + j] / fVal;
		}
	}
}

Matd operator+(const Matd& fA, const Matd& fB)
{
	Matd out(fA.rows, fA.cols);
	int collOff;
	for(int i = 0; i < fA.cols; i++)
	{
		collOff = i * fA.rows;
		for(int j = 0; j < fA.rows; j++)
		{
			out.data[collOff + j] = fA.data[collOff + j] + fB.data[collOff + j];
		}
	}	
	return out;
}

Matd operator-(const Matd& fA, const Matd& fB)
{
	Matd out(fA.rows, fA.cols);
	int collOff;
	for(int i = 0; i < fA.cols; i++)
	{
		collOff = i * fA.rows;
		for(int j = 0; j < fA.rows; j++)
		{
			out.data[collOff + j] = fA.data[collOff + j] - fB.data[collOff + j];
		}
	}	
	return out;
}

void Matd::operator+=(const Matd& fMat)
{
	int collOff;
	for(int i = 0; i < fMat.cols; i++)
	{
		collOff = i * fMat.rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			data[collOff + j] += fMat.data[collOff + j];
		}
	}
}

void Matd::operator-=(const Matd& fMat)
{
	int collOff;
	for(int i = 0; i < fMat.cols; i++)
	{
		collOff = i * fMat.rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			data[collOff + j] -= fMat.data[collOff + j];
		}
	}	
}

Matd HadProd(const Matd& fA, const Matd& fB)
{
	Matd out(fA.rows, fA.cols);
	int collOff;
	for(int i = 0; i < fA.cols; i++)
	{
		collOff = i * fA.rows;
		for(int j = 0; j < fA.rows; j++)
		{
			out.data[collOff + j] = fA.data[collOff + j] * fB.data[collOff + j];
		}
	}	
	return out;
}

Matd Trans(const Matd& fA)
{
	Matd out(fA.cols, fA.rows);
	int collOff;
	for(int i = 0; i < fA.cols; i++)
	{
		collOff = i * fA.rows;
		for(int j = 0; j < fA.rows; j++)
		{
			out.data[i + out.rows * j] = fA.data[collOff + j];
		}
	}
	return out;
}

Matd MultTransA(const Matd& fA, const Matd& fB)
{
	//A B
	int r = fA.rows;
	int c = fB.cols;
	Matd out(r, c);
		
#ifndef CBLAS
	Matd a = Trans(fA);
	int colOffOut; 
	int colOffsetB;
	double sum;
	for(int i = 0; i < c; i++)
	{
		colOffOut = i * r;
		colOffsetB = i * fB.rows;
		for(int j = 0; j < r; j++)
		{
			sum = 0; 
			for(int k = 0; k < fB.rows; k++)
			{
				sum += a.data[k * r + j] * fB.data[colOffsetB + k];
			}
			out.data[colOffOut + j] = sum;
		}
	}
#else
	CBLAS_ORDER ord = CblasColMajor;
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	CBLAS_TRANSPOSE transTrue = CblasTrans;
	cblas_dgemm(ord, transTrue, trans, fA.rows, fB.cols, fB.rows, 1.0, fA.data, fA.cols, fB.data, fB.rows, 0, out.data, fA.rows);
#endif

	return out;
}

Matd OuterProduct(const Matd& fA, const Matd& fB)
{
	int aColOff; 
	int outOff;
	int r = fA.rows;
	int c = fB.cols;
	Matd out(r, c);
	//fA.cols == fB.rows;
	for(int i = 0; i < fA.cols; i++)
	{
		aColOff = i * fA.rows;
		for(int j = 0; j < c; j++)
		{
			outOff = j * fA.rows;
			for(int k = 0; k < r; k++)
			{
				out.data[outOff + k] += fA.data[aColOff + k] * fB.data[i + j * fB.rows];
			}
		}
	}
	return out;
}

void Matd::Print()
{	
	printf("\n rows = %d, cols = %d\n", rows, cols);
	for(int i = 0; i < rows; i++)
	{
		printf("|");
		for(int j = 0; j < cols; j++)
		{
			printf("%4.4f,", Get(i, j));
		}
		printf("|\n");
	}
}

void Matd::Randomize(double fMin, double fMax)
{
	int collOff;
	double range = fMax - fMin;
	for(int i = 0; i < cols; i++)
	{
		collOff = i * rows;
		for(int j = 0; j < rows; j++)
		{
			data[collOff + j] = (range * ((float)rand() / RAND_MAX)) + fMin;
		}
	}
}

void Matd::SetAll(double fVal)
{
	int collOff;
	for(int i = 0; i < cols; i++)
	{
		collOff = i * rows;
		for(int j = 0; j < rows; j++)
		{
			data[collOff + j] = fVal;
		}
	}
}

void Matd::SetRow(int fRow, double fVal)
{
	int collOff = 0;
	for(int i = 0; i < cols; i++)
	{
		//collOff = i * rows;
		data[collOff + fRow] = fVal;
		collOff += rows;
	}
}

void Matd::SetCol(int fCol, double fVal)
{
	int collOff = fCol * rows;
	for(int i = 0; i < rows; i++)
	{
		data[collOff + i] = fVal;
	}
}

void Matd::ComponentFunction(double (*Func)(double a))
{
	int collOff;
	for(int i = 0; i < cols; i++)
	{
		collOff = i * rows;
		for(int j = 0; j < rows; j++)
		{
			data[collOff + j] = Func(data[collOff + j]);
		}
	}
}

void Matd::Copy(Matd& fMat, int fRowOffset, int fColOffset)
{
	for(int i = 0; i < fMat.cols; i++)
	{
		int copyColOff = i * fMat.rows;
		int colOff = (i + fColOffset) * rows;
		for(int j = 0; j < fMat.rows; j++)
		{
			data[colOff + j + fRowOffset] = fMat.data[copyColOff + j];
		}
	}
}





class NN
{
	public:
		static double FastExp(double x);
		static double SigmiodActivationFunction(double x);
		static double GradientSigmiodActivationFunction(double x);
	
		int layerNumber;
		Matd* layer;
		Matd output;
		
		NN();
		NN(int fInputNumber, int fLayerNumber, const int fLayerSizes[]);
		~NN();
		void Create(int fInputNumber, int fLayerNumber, const int fLayerSizes[]);
		void Destroy();
		void ForwardUpdate(Matd fInput);
		void Print();
		void BackPropogate(Matd& fInput, Matd& fTrainingOutput, double fLearningRate);
};

double NN::FastExp(double x) 
{
  x = 1.0 + x / 1024;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x;
  return x;
}

double NN::SigmiodActivationFunction(double x)
{
	return 1.0 / (1.0 + FastExp(-x));
}

double NN::GradientSigmiodActivationFunction(double x)
{
	double sigm = 1.0 / (1.0 + FastExp(-x));
	return (1 - sigm) * sigm;
}

NN::NN()
{
	layerNumber = 0;
}

NN::NN(int fInputNumber, int fLayerNumber, const int fLayerSizes[])
{
	layerNumber = 0;
	Create(fInputNumber, fLayerNumber, fLayerSizes);
}

NN::~NN()
{
	Destroy();
}

void NN::Create(int fInputNumber, int fLayerNumber, const int fLayerSizes[])
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

void NN::Destroy()
{
	if(layerNumber != 0)
	{
		layerNumber = 0;
		delete[] layer;
	}
	output.Destroy();
}

void NN::ForwardUpdate(Matd fInput)
{
	Matd aRun(fInput.rows + 1, fInput.cols);
	aRun.SetRow(fInput.rows, 1.0);
	aRun.Copy(fInput, 0, 0);
	output = aRun;
	for(int i = 0; i < layerNumber; i++)
	{
		output = layer[i] * output;
		output.ComponentFunction(SigmiodActivationFunction);
		output.SetRow(output.rows - 1, 1.0);
	}
}

void NN::Print()
{
	for(int i = 0; i < layerNumber; i++)
	{
		printf("layer %d:\n", i);
		layer[i].Print();
	}
}

void NN::BackPropogate(Matd& fInput, Matd& fTrainingOutput, double fLearningRate)
{
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
	
	for(int i = 0; i < layerNumber; i++)
	{
		aRun = layer[i] * aRun;
		z[i] = aRun;
		aRun.ComponentFunction(SigmiodActivationFunction);
		a[i] = aRun;
		aRun.SetRow(aRun.rows - 1, 1.0);
	}
	//Compute output error
	int L = layerNumber - 1;
	z[L].ComponentFunction(GradientSigmiodActivationFunction);
	z[L].SetRow(z[L].rows - 1, 1.0);
	delta[L] = HadProd(a[L] - y, z[L]);
	layer[L] -= OuterProduct(delta[L], Trans(a[L])) * (fLearningRate / fInput.cols);
	//Back Propagate the error
	for(int i = L - 1; i >= 0; i--)
	{
		z[i].ComponentFunction(GradientSigmiodActivationFunction);
		z[i].SetRow(z[i].rows - 1, 1.0);
		delta[i] = HadProd(MultTransA(layer[i + 1], delta[i + 1]), z[i]);
		//Gradient decent to adjust weights
		layer[i] -= OuterProduct(delta[i], Trans(a[i])) * (fLearningRate / fInput.cols);
	}
}

int main()
{
	const int layersize[] = {1};
	clock_t t; t = clock();
	
	int tSetSize = 2;
	Matd tin(1, tSetSize);
	for(int i = 0; i < tSetSize; i++)
		tin.Set(0, i, i);
		
	Matd tout(1, tSetSize);
	for(int i = 0; i < tSetSize; i++)
		tout.Set(0, i, i);
		
	NN nn(1, 1, layersize);
	nn.BackPropogate(tin, tout, 100);
	
	Matd in(1, 2);
	in.Set(0, 0, 0);
	in.Set(0, 1, 1);
	nn.ForwardUpdate(in);
	in.Print();
	nn.layer[0].Print();
	nn.output.Print();
	
	t = clock() - t; printf("took %f s\n", ((float)t) / CLOCKS_PER_SEC);
	printf("done nn test 1\n");
	
	return 0;
}