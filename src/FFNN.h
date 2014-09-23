#ifndef FFNN_H
#define FFNN_H

#include "matd.h"

class FFNN
{		
	private:
		int layerNumber;
		Matd* layer;
		
	public:
		static double FastExp(double fX);
		static double Sigmiod(double fX);
		static double GradientSigmiod(double fX);
	
		FFNN();
		~FFNN();
		void Create(int fInputNumber, int fLayerNumber, const int fLayerSizes[]);
		void Destroy();
		Matd ForwardUpdate(Matd& fInput);
		void Print();
		void BackPropogate(Matd& fInput, Matd& fTrainingOutput, double fLearningRate);
		void Save(const char* fFN);
		void Load(const char* fFN);
};

#endif