#include <stdlib.h>
#include "matrixSaver.h"

int main(int argc, char* argv[])
{
	Matd mat(4, 3);
	mat.Set(0, 0, 1);
	mat.Set(1, 0, 2);
	mat.Set(2, 0, 3);
	mat.Set(3, 0, 4);
	mat.Set(0, 1, 5);
	mat.Set(1, 1, 6);
	mat.Set(2, 1, 7);
	mat.Set(3, 1, 8);
	mat.Set(0, 2, 9);
	mat.Set(1, 2, 10);
	mat.Set(2, 2, 11);
	mat.Set(3, 2, 12);
	
	MatrixSaver saver("testFile.csv");
	saver.WriteMatrix(mat);
	
	return 0;
}