#pragma once

namespace NN
{
	
namespace Util
{
	

/*
 *
 * Filp Bytes Function
 *
 */
 
 
inline void FilpBytes(void* input, int byteNum)
{
	for(int i = 0; i < byteNum / 2; i++)
	{
		uint8_t temp = ((uint8_t*)input)[i];
		((uint8_t*)input)[i] = ((uint8_t*)input)[byteNum - i - 1];
		((uint8_t*)input)[byteNum - i - 1] = temp;
	}
}

}

}
