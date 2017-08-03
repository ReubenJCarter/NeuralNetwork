#pragma once


#include "CommonHeaders.h"


namespace NN
{	

namespace Util
{
	

/*
 *
 * MNISTImageFile
 *
 */

 
class MNISTImageFile
{
	private:
		uint32_t magicNumber;
		uint32_t imageNumber;
		uint32_t height;
		uint32_t width;
		std::ifstream fileS;
		bool isOpen;
		
	public:
		MNISTImageFile();
		bool Open(const char* fileName);
		int GetImageNumber();
		int GetWidth();
		int GetHeight();
		bool GetImageData(int imageIndex, std::vector<uint8_t>& data);
		bool GetImageData(int imageIndex, void* data);
		bool Close();
};

	
	
}
}
