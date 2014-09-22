#include "matd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef CBLAS
#include <cblas.h>
#endif

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
	return out;
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
	#ifndef CBLAS
	out.SetAll(0);
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
	#else
	out = fA * fB;
	#endif
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