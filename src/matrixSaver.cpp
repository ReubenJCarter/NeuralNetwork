#include "matrixSaver.h"

MatrixSaver::MatrixSaver()
{
	
}

MatrixSaver::MatrixSaver(const char* fFN)
{
	OpenCSVFile(fFN);
}

MatrixSaver::~MatrixSaver()
{
	CloseFile();
}

void MatrixSaver::OpenCSVFile(const char* fFN)
{
	fptr = fopen(fFN, "wb");
	if(fptr == NULL) printf("failed to open a csv file\n");
}

void MatrixSaver::CloseFile()
{
	fclose(fptr);
}

void MatrixSaver::WriteMatrix(Matd& fMat)
{
	for(int i = 0; i < fMat.rows; i++)
	{
		for(int j = 0; j < fMat.cols; j++)
		{
			fprintf(fptr, "%lf", fMat.Get(i, j));
			if(j < fMat.cols - 1)
				fprintf(fptr, ",");
			else
				fprintf(fptr, "\n");
		}
	}
}