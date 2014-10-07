#include "csvFileParser.h"
#include <stdlib.h>

int CSVFileParser::GetNextChar()
{
	return fgetc(fptr);
}

CSVFileParser::CSVFileParser()
{
	fptr = 0;
	currentCol = 0;
	currentRow = 0;
}

CSVFileParser::CSVFileParser(const char* fFN)
{
	Open(fFN);
	currentCol = 0;
	currentRow = 0;
}

void CSVFileParser::Open(const char* fFN)
{
	
	fptr = fopen(fFN, "rb");
	if(fptr == NULL) printf("failed to open a csv file\n");
}

void CSVFileParser::Close()
{
	fclose(fptr);
	currentCol = 0;
	currentRow = 0;
}

int CSVFileParser::Get(char* fBuffer, int fRow, int fCol)
{
	int len = 0; 
	int nextChar = GetNextChar();
	int found = 0;
	
	if(fRow < currentRow || (fRow == currentRow && fCol < currentCol))
	{
		fseek(fptr, 0, SEEK_SET);
		currentRow = 0;
		currentCol = 0;
	}
	
	while(!found)
	{
		while(nextChar != ',' && nextChar != '\n' && nextChar != EOF)
		{
			if(fCol == currentCol && fRow == currentRow) 
			{
				fBuffer[len] = (char)nextChar;
				found = 1;
				len++;
			}
			nextChar = GetNextChar();
		}
		if(nextChar == ',')
		{
			currentCol++;
			nextChar = GetNextChar();
		}
		else if(nextChar == '\n')
		{
			currentCol = 0;
			currentRow++;
			nextChar = GetNextChar();
		}
		else if(nextChar == EOF)
		{
			break;
		}
		if(currentRow > fRow)
		{
			break;
		}
	}

	return len;
}

int CSVFileParser::GetString(char* fBuffer, int fRow, int fCol)
{
	int len = Get(fBuffer, fRow, fCol);
	fBuffer[len] = '\0';
	return len + 1;
}

int CSVFileParser::GetInt(int fRow, int fCol)
{
	char buffer[100];
	Get(buffer, fRow, fCol);
	return atoi(buffer);
}

double CSVFileParser::GetDouble(int fRow, int fCol)
{
	char buffer[100];
	Get(buffer, fRow, fCol);
	return atof(buffer);	
}
