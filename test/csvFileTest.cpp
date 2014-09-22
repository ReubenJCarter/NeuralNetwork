#include <stdlib.h>
#include "csvFile.h"

const char* csvFile = "a,1,10,100\n"
					  "b,3,30,300\n"
					  "c,5,50,500\n"
					  "d,7,70,700";

int main(int argc, char* argv[])
{
	FILE* fptr;
	fptr = fopen("testFile.csv", "wb");
	fprintf(fptr, "%s",csvFile);
	fclose(fptr);
	int row = 0;
	int col = 0;
	CSVFile csv("testFile.csv");
	char buffer[1000];
	int len = 2;
	
	len = csv.GetString(buffer, 2, 1);
	printf("(2,1):%s\n", buffer);
	len = csv.GetString(buffer, 1, 1);
	printf("(1,1):%s\n", buffer);
	len = csv.GetString(buffer, 0, 1);
	printf("(0,1):%s\n", buffer);
	
	len = csv.GetString(buffer, 1, 2);
	printf("(1,2):%s\n", buffer);
	len = csv.GetString(buffer, 1, 1);
	printf("(1,1):%s\n", buffer);
	len = csv.GetString(buffer, 1, 0);
	printf("(1,0):%s\n", buffer);
	
	return 0;
}