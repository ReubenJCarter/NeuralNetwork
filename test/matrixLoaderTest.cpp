#include <stdio.h>
#include "fileAccess/matrixLoader.h"

const char* csvFile = 
"Date,Open,High,Low,Close,Volume,Adj Close\n"
"2014-08-18,587.10,595.05,586.55,592.70,1466000,592.70\n"
"2014-08-15,587.50,589.47,580.76,583.71,1723400,583.71\n"
"2014-08-14,586.69,587.80,580.92,584.65,1272900,584.65\n"
"2014-08-13,576.45,586.13,575.20,584.56,1903300,584.56\n"
"2014-08-12,575.00,575.90,569.91,572.12,1394500,572.12\n"
"2014-08-11,579.00,579.69,575.30,577.25,1171300,577.25\n"
"2014-08-08,572.02,579.56,569.02,577.94,1493300,577.94\n"
"2014-08-07,576.05,578.31,569.43,571.81,1163000,571.81\n"
"2014-08-06,569.50,578.64,567.45,574.49,1322800,574.49\n"
"2014-08-05,579.38,580.20,570.31,573.14,1643800,573.14\n"
"2014-08-04,576.51,583.82,572.26,582.27,1519400,582.27\n"
"2014-08-01,578.55,583.43,570.30,573.60,2213300,573.60\n"
"2014-07-31,588.96,591.99,577.68,579.55,2309500,579.55\n"
"2014-07-30,595.81,598.45,592.70,595.44,1215100,595.44\n"
"2014-07-29,597.70,598.49,592.17,593.95,1366600,593.95";

int main()
{
	FILE* fptr;
	fptr = fopen("testFile.csv", "wb");
	fprintf(fptr, "%s",csvFile);
	fclose(fptr);
	
	int dataPoints = 4;
	Matd mat(4, dataPoints);
	MatrixLoader loader;
	loader.OpenCSVFile("testFile.csv");
	loader.ReadRowFromVertical(mat, 0, 1, 1, dataPoints);
	loader.ReadRowFromVertical(mat, 1, 1, 2, dataPoints);
	loader.ReadRowFromVertical(mat, 2, 1, 3, dataPoints);
	loader.ReadRowFromVertical(mat, 3, 1, 5, dataPoints);
	mat.Print();
	
	return 0;
}