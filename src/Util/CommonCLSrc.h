#pragma once 

#include "CommonHeaders.h"

namespace NN
{

	
//
//Activation Function 
//

const std::string activationFunctionsSrc = ""

//
//Activation Functions
//

"float ActivationIdentity(float v, float param0, float param1)"
"{"
"	return v;"
"}"
"float DeltaActivationIdentity(float v, float param0, float param1)"
"{"
"	return 1;"
"}"
""
"float ActivationBinaryStep(float v, float param0, float param1)"
"{"
"	return v < 0 ? 0 : 1;"
"}"
"float DeltaActivationBinaryStep(float v, float param0, float param1)"
"{"
"	return 0;"
"}"
""
"float ActivationLogistic(float v, float param0, float param1)"
"{"
"	return 1.0 /(1.0 + exp(-v));"
"}"
"float DeltaActivationLogistic(float v, float param0, float param1)"
"{"
"	float f = 1.0 / (1.0 + exp(-v));"
"	return f * (1.0 - f);"
"}"
""
"float ActivationTanH(float v, float param0, float param1)"
"{"
"	return tanh(v);"
"}"
"float DeltaActivationTanH(float v, float param0, float param1)"
"{"
"	float f = tanh(v);"
"	return 1.0 - f * f;"
"}"
""
"float ActivationArcTan(float v, float param0, float param1)"
"{"
"	return atan(v);"
"}"
"float DeltaActivationArcTan(float v, float param0, float param1)"
"{"
"	return 1.0/(v * v + 1);"
"}"
""
"float ActivationSoftSign(float v, float param0, float param1)"
"{"
"	return v / (1.0 + fabs(v));"
"}"
"float DeltaActivationSoftSign(float v, float param0, float param1)"
"{"
"	float f = 1.0 + fabs(v);"
"	return 1.0 / (f * f);"
"}"
""
"float ActivationReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? 0 : v;"
"}"
"float DeltaActivationReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? 0 : 1;"
"}"
""
"float ActivationLeakyReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? 0.01 * v : v;"
"}"
"float DeltaActivationLeakyReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? 0.01 : 1;"
"}"
""
"float ActivationLeakyPReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? param0 * v : v;"
"}"
"float DeltaActivationLeakyPReLU(float v, float param0, float param1)"
"{"
"	return v < 0 ? param0 : 1;"
"}"
""
"float ActivationELU(float v, float param0, float param1)"
"{"
"	return v < 0 ? param0 * (exp(v) - 1.0) : v;"
"}"
"float DeltaActivationELU(float v, float param0, float param1)"
"{"
"	float f = param0 * (exp(v) - 1.0);"
"	return v < 0 ? f + param0 : 1;"
"}"
""
"float ActivationSoftPlus(float v, float param0, float param1)"
"{"
"	return log(1.0 + exp(v));"
"}"
"float DeltaActivationSoftPlus(float v, float param0, float param1)"
"{"
"	return 1.0 / (1.0 + exp(-v));"
"}"
""
"float ActivationGaussian(float v, float param0, float param1)"
"{"
"	return exp(-v * v);"
"}"
"float DeltaActivationGaussian(float v, float param0, float param1)"
"{"
"	return -2 * v * exp(-v * v);"
"}"
""
""
""
"typedef enum {IDENTITY, BINARY_STEP, LOGISTIC} ACTIVATION_TYPE;"
"float ActivationFunction(float v, float param0, float param1, ACTIVATION_TYPE activationType)"
"{"
"	float x;"

"	if(activationType == IDENTITY)"
"		x = ActivationIdentity(v, param0, param1);"

"	else if(activationType == BINARY_STEP)"
"		x = ActivationBinaryStep(v, param0, param1);"

"	else if(activationType == LOGISTIC)"
"		x = ActivationLogistic(v, param0, param1);"

"	return x;"
"}"
"float DeltaActivationFunction(float v, float param0, float param1, ACTIVATION_TYPE activationType)"
"{"
"	float x;"

"	if(activationType == IDENTITY)"
"		x = DeltaActivationIdentity(v, param0, param1);"

"	else if(activationType == BINARY_STEP)"
"		x = DeltaActivationBinaryStep(v, param0, param1);"

"	else if(activationType == LOGISTIC)"
"		x = DeltaActivationLogistic(v, param0, param1);"

"	return x;"
"}"


//
//Cost Functions
//


""
"float DeltaCostQuadratic(float activation, float trainExample)"
"{"
"	return activation - trainExample;"
"}"
"float DeltaCostCrossEntropy(float activation, float trainExample)"
"{"
"	return (activation - trainExample) / ((1 - activation) * activation);"
"}"
""
""
""
"typedef enum {QUADRATIC, CROSS_ENTROPY} COST_TYPE;"
"float DeltaCostFunction(float activation, float trainExample, COST_TYPE costType)"
"{"
"	float x;"

"	if(costType == QUADRATIC)"
"		x = DeltaCostQuadratic(activation, trainExample);"

"	else if(costType == CROSS_ENTROPY)"
"		x = DeltaCostCrossEntropy(activation, trainExample);"

"	return x;"
"}"
;

}