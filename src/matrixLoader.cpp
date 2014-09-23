#include "matrixLoader.h"

MatrixLoader::MatrixLoader()
{
	
}

MatrixLoader::~MatrixLoader()
{
	csvFile.Close();
}

void MatrixLoader::OpenCSVFile(const char* fFN)
{
	csvFile.Open(fFN);
}

void MatrixLoader::ReadRowFromVertical(Matd& fMat, int fRow, int fStartReadRow, int fStartReadCol, int fSteps)
{
	for(int i = 0; i < fSteps; i++)
	{
		double val = csvFile.GetDouble(fStartReadRow + i, fStartReadCol);
		fMat.Set(fRow, i, val);
	}
}

void MatrixLoader::ReadRowFromHorizontal(Matd& fMat, int fRow, int fStartReadRow, int fStartReadCol, int fSteps)
{
	for(int i = 0; i < fSteps; i++)
	{
		double val = csvFile.GetDouble(fStartReadRow, fStartReadCol + i);
		fMat.Set(fRow, i, val);
	}
}

void MatrixLoader::ReadColFromVertical(Matd& fMat, int fCol, int fStartReadRow, int fStartReadCol, int fSteps)
{
	for(int i = 0; i < fSteps; i++)
	{
		double val = csvFile.GetDouble(fStartReadRow + i, fStartReadCol);
		fMat.Set(i, fCol, val);
	}
}

void MatrixLoader::ReadColFromHorizontal(Matd& fMat, int fCol, int fStartReadRow, int fStartReadCol, int fSteps)
{
	for(int i = 0; i < fSteps; i++)
	{
		double val = csvFile.GetDouble(fStartReadRow, fStartReadCol + i);
		fMat.Set(i, fCol, val);
	}
}