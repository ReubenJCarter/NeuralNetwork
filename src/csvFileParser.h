#ifndef CSVFILEPARSER_H
#define CSVFILEPARSER_H

#include <stdio.h>

class CSVFileParser
{
	private:
		FILE* fptr;
		int currentRow;
		int currentCol;
		
		int GetNextChar();
		
	public:
		CSVFileParser();
		CSVFileParser(const char* fFN);
		void Open(const char* fFN);
		void Close();
		int Get(char* fBuffer, int fRow, int fCol);
		int GetString(char* fBuffer, int fRow, int fCol);
		int GetInt(int fRow, int fCol);
		double GetDouble(int fRow, int fCol);
};

#endif