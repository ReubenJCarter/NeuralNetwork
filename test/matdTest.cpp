#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matd.h"

double CompFunc(double a)
{
	return a / 10.0;
}

int main()
{
	//*******************//
	//MULTIPLICATION TEST//
	//*******************//
	
	printf("//*******************//\n//MULTIPLICATION TEST//\n//*******************//\n");
	Matd a(2, 2);
	Matd b(2, 2);
	a.Set(0, 0, 1);
	a.Set(0, 1, 2);
	a.Set(1, 0, 3);
	a.Set(1, 1, 4);
	b.Set(0, 0, 5);
	b.Set(0, 1, 6);
	b.Set(1, 0, 7);
	b.Set(1, 1, 8);	
	Matd c;
	c = a * b;
	printf("a:\n");
	a.Print();
	printf("b:\n");
	b.Print();
	printf("matrix c is matrix a multiplied by matrix b:\n");
	c.Print();
	printf("matrix d is 10 * c * 10:\n");
	Matd d; 
	d = 10 * c * 10;
	d.Print();
	printf("matrix e is d / 10:\n");
	Matd e; 
	e = d / 10;
	e.Print();
	printf("END\n");
	
	//*************************//
	//ADDITION SUBTRACTION TEST//
	//*************************//

	printf("//*************************//\n//ADDITION/SUBTRACTION TEST//\n//*************************//\n");
	c = a + b;
	printf("a:\n");
	a.Print();
	printf("b:\n");
	b.Print();
	printf("matrix c is matrix a added to matrix b:\n");
	c.Print();
	printf("matrix d is matrix a subtract matrix b:\n");
	d = a - b;
	d.Print();
	printf("matrix d is += b:\n");
	d += b;
	d.Print();
	printf("matrix d is -= b:\n");
	d -= b;
	d.Print();
	printf("END\n");

	//********************************//
	//HAD, TRANS, MULTTRANS, OUTERPROD//
	//********************************//

	printf("//********************************//\n//HAD, TRANS, MULTTRANS, OUTERPROD TEST//\n//********************************//\n");
	printf("a:\n");
	a.Print();
	printf("b:\n");
	b.Print();	

	d = HadProd(a, b);
	printf("matrix d is matrix a hadProd matrix b:\n");
	d.Print();
	
	d = Trans(a);
	printf("matrix d is trans a:\n");
	d.Print();
	
	d = MultTransA(a, b);
	printf("matrix d is matrix Trans(a) multiplied by matrix b:\n");
	d.Print();
	
	Matd x(2, 1);
	Matd y(1, 2);
	x.Set(0, 0, 2);
	x.Set(1, 0, 4);
	y.Set(0, 0, 5);
	y.Set(0, 1, 6);
	printf("x and y:\n");
	x.Print();
	y.Print();
	d = x * y;
	printf("d, outerproduct x and y:\n");
	d.Print();
	
	//************//
	//Ranndom Test//
	//************//
	
	printf(	"//************//\n//Ranndom Test//\n//************//\n");
	Matd z(10, 10);
	printf("z is a completely random matrix:\n");
	z.Randomize(0.0, 1.0);
	z.Print();
	printf("z has been set to 1.321:\n");
	z.SetAll(1.321);
	z.Print();
	printf("z row 3 has been set to 9:\n");
	z.SetRow(3, 9);
	z.Print();
	printf("z col 4 has been set to 8:\n");
	z.SetCol(4, 8);
	z.Print();
	printf("z has been given a function to divide all cells by 10:\n");
	z.ComponentFunction(CompFunc);
	z.Print();
	
	//*****//
	//Speed//
	//*****//
	printf(	"//*****//\n//Speed//\n//*****//\n");
	int i = 1000;
	int j = 1000;
	Matd bigMat0(i, j);
	Matd bigMat1(i, j);
	Matd resultMat;
	bigMat0.Randomize(0.0, 1.0);
	bigMat1.Randomize(0.0, 1.0);
	clock_t t;
	printf("computing matrix multiplication of two %d by %d matricies:\n", i, j);
	t = clock();
	resultMat = bigMat0 * bigMat1;
	t = clock() - t;
	printf("completed in %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
	
	return 0;
}

