#ifndef MATRIXLOADER_H
#define MATRIXLOADER_H

#include "csvFile.h"
#include "matd.h"

class MatrixFile
{
	private:
		CSVFile csvFile;
		
	public:
		MatrixFile();
		Matd Load();
		void Save(Matd fM);
};

#endif