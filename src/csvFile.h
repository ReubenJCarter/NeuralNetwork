#ifndef CSVFILE_H
#define CSVFILE_H

#include <stdio.h>

class CSVFile
{
	private:
		FILE* fptr;
		int currentRow;
		int currentCol;
		
		int GetNextChar();
		
	public:
		CSVFile();
		CSVFile(const char* fFN);
		void Open(const char* fFN);
		void Close();
		int Get(char* fBuffer, int fRow, int fCol);
		int GetString(char* fBuffer, int fRow, int fCol);
		int GetInt(int fRow, int fCol);
		double GetDouble(int fRow, int fCol);
};

#endif