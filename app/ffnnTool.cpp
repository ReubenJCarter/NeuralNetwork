#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "FFNN.h"
#include "matd.h"
#include "matrixLoader.h"
#include "matrixSaver.h"

#define MAX_LAYERS 1000

class Clp
{
	public:
		Clp(int argc, char *argv[]);
		
};

void CreateTool(int argc, char *argv[])
{

}

int main(int argc, char *argv[])
{
	if(argc > 1)
	{
		if(strcmp(argv[1], "create") == 0)
		{
			CreateTool(argc, argv);
		}
		else if(strcmp(argv[1], "train") == 0)
		{
			
		}
		else if(strcmp(argv[1], "run") == 0)
		{
			
		}
	}
	else
	{
		printf("not enough arguments.");
	}
	
	return 0;
}