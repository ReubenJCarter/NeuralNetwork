#ifndef FFNN_H
#define FFNN_H

#include "matd.h"

class FFNN
{		
	public:
		typedef double (*CompFunction)(double);
		
	private:
		int layerNumber;
		Matd* layer;
		CompFunction* activationFunction;
		CompFunction* gradientActivationFunction;
		
		static double Square(double fX);
		
	public:		
		static double FastExp(double fX);
		static double Sigmiod(double fX);
		static double GradientSigmiod(double fX);
		static double Linear(double fX);
		static double GradientLinear(double fX);
	
		FFNN();
		~FFNN();
		void Create(int fInputNumber, int fLayerNumber, const int fLayerSizes[]);
		void SetActivationFunction(int fLayer, CompFunction fActivationFunction, CompFunction fGradientActivationFunction);
		void Destroy();
		Matd ForwardUpdate(Matd& fInput);
		void Print();
		void BackPropogate(Matd& fInput, Matd& fTrainingOutput, double fLearningRate);
		double GetCost(Matd& fInput, Matd& fTrainingOutput);
		void Save(const char* fFN);
		void Load(const char* fFN);
		void SetWeight(int fLayer, int fNeuron, int fWeight, double fValue);
};

#endif