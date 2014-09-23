#ifndef MATRIXLOADER_H
#define MATRIXLOADER_H

#include "csvFile.h"
#include "matd.h"

class MatrixLoader
{
	private:
		CSVFile csvFile;
		
	public:
		MatrixLoader();
		~MatrixLoader();
		void OpenCSVFile(const char* fFN);
		void ReadRowFromVertical(Matd& fMat, int fRow, int fStartReadRow, int fStartReadCol, int fSteps);
		void ReadRowFromHorizontal(Matd& fMat, int fRow, int fStartReadRow, int fStartReadCol, int fSteps);
		void ReadColFromVertical(Matd& fMat, int fCol, int fStartReadRow, int fStartReadCol, int fSteps);
		void ReadColFromHorizontal(Matd& fMat, int fCol, int fStartReadRow, int fStartReadCol, int fSteps);
};

#endif