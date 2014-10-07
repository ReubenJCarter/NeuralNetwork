#ifndef MATRIXSAVER_H
#define MATRIXSAVER_H

#include "matd.h"
#include <stdio.h>

class MatrixSaver
{
	private:
		FILE* fptr;
		
	public:
		MatrixSaver();
		MatrixSaver(const char* fFN);
		~MatrixSaver();
		void OpenCSVFile(const char* fFN);
		void CloseFile();
		void WriteMatrix(Matd& fMat);
};

#endif