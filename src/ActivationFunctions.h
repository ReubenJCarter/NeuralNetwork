#pragma once


#include <math.h>


namespace NN
{


//
//Activation Function 
//

class ActivationFunction
{
public:

	virtual float F(float x);
	virtual float DF(float x);
}; 


//
//Activation Function 
//


class IdentityFunction:public ActivationFunction
{
public:

	inline virtual float F(float x) final{return x;}
	inline virtual float DF(float x) final{return 1;};


};


//
//BinaryStep Function 
//


class BinaryStep:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return x < 0 ? 0 : 1;
	}
	
	inline virtual float DF(float x) final
	{
		return 0;
	}
};


//
//LogisticFunction Function 
//


class LogisticFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return 1.0 /(1.0 + exp(-x));
	}
	
	inline virtual float DF(float x) final
	{
		float f = 1.0 /(1.0 + exp(-x));
		return f * (1.0 - f);
	}
};


//
//TanH Function 
//


class TanHFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return tanh(x);
	}
	
	inline virtual float DF(float x) final
	{
		float f =  tanh(x);
		return 1.0 - f * f
	}
};


//
//ArcTan Function 
//


class ArcTanFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return atan(x);
	}
	
	inline virtual float DF(float x) final
	{
		return 1.0/(x * x + 1);
	}
};


//
//SoftSign Function 
//


class SoftSignFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return x / (1 + abs(x));
	}
	
	inline virtual float DF(float x) final
	{
		float t = 1 + abs(x);
		return 1.0 / (t * t);
	}
};


//
//ReLU Function 
//


class ReLUFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return x < 0 ? 0 : x;
	}
	
	inline virtual float DF(float x) final
	{
		return x < 0 ? 0 : 1;
	}
};


//
//LeakyReLU Function 
//


class LeakyReLUFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return x < 0 ? 0.01 * x : x;
	}
	
	inline virtual float DF(float x) final
	{
		return x < 0 ? 0.01 : 1;
	}
};


//
//PReLU Function 
//


class PReLUFunction:public ActivationFunction
{
public:
	float a;
	
	PReLUFunction():a(0.01){};

	inline virtual float F(float x) final
	{
		return x < 0 ? a * x : x;
	}
	
	inline virtual float DF(float x) final
	{
		return x < 0 ? a : 1;
	}
};


//
//ELU Function 
//


class ELUFunction:public ActivationFunction
{
public:
	float a;
	
	ELUFunction():a(0.5){};

	inline virtual float F(float x) final
	{
		return x < 0 ? a * (exp(x) - 1.0) : x;
	}
	
	inline virtual float DF(float x) final
	{
		float f = a * (exp(x) - 1.0);
		return x < 0 ? f + a : 1;
	}
};


//
//SoftPlus Function 
//


class SoftPlusFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return log(1.0 + exp(x)); //log is ln() in math.h
	}
	
	inline virtual float DF(float x) final
	{
		return 1.0 / (1.0 + exp(-x)); 
	}
};


//
//Gaussian Function 
//


class GaussianFunction:public ActivationFunction
{
public:
	inline virtual float F(float x) final
	{
		return exp(-x * x);
	}
	
	inline virtual float DF(float x) final
	{
		return -2 * x * exp(-x * x);
	}
};


}