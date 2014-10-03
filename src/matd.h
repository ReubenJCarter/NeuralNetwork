#ifndef MATD_H
#define MATD_H

class Matd
{
	public:
		int rows;
		int cols;
		int size;
		double* data;
		
		Matd();
		Matd(int fRow, int fCol);
		~Matd();
		double Get(int fRow, int fCol);
		void Set(int fRow, int fCol, double fVal);
		void Create(int fRow, int fCol);
		void Destroy();
		void operator=(const Matd& fMat);
		friend Matd operator*(const Matd& fA, const Matd& fB);
		friend Matd operator*(const Matd& fMat, const double fVal);
		friend Matd operator*(const double fVal, const Matd& fMat);
		friend Matd operator/(const Matd& fMat, const double fVal);
		friend Matd operator+(const Matd& fA, const Matd& fB);
		friend Matd operator-(const Matd& fA, const Matd& fB);
		void operator+=(const Matd& fMat);
		void operator-=(const Matd& fMat);
		friend Matd HadProd(const Matd& fA, const Matd& fB);
		friend Matd Trans(const Matd& fA);
		friend Matd MultTransA(const Matd& fA, const Matd& fB);
		friend Matd OuterProduct(const Matd& fA, const Matd& fB);
		void Print();
		void Randomize(double fMin, double fMax);
		void SetAll(double fVal);
		void SetRow(int fRow, double fVal);
		void SetCol(int fCol, double fVal);
		void ComponentFunction(double (*Func)(double a));
		void Copy(Matd& fMat, int fRowOffset, int fColOffset);
		void Copy(Matd& fMat, int fFromRowOffset, int fFromColOffset, int fToRowOffset, int fToColOffset, int fRows, int fCols);
};

#endif