#pragma once


#include "CommonHeaders.h"


namespace NN
{	

namespace Util
{
	

/*
 *
 * MNISTLabelFile class
 *
 */


class MNISTLabelFile
{
	private:
		uint32_t magicNumber;
		uint32_t labelNumber;
		std::ifstream fileS;
		bool isOpen;
		bool dataPreloaded;
		std::vector<uint8_t> preloadedData;
		
	public:
		MNISTLabelFile();
		bool Open(const char* fileName, bool preload=true);
		int GetLabelNumber();
		bool GetLabelData(int labelIndex, uint8_t& data);
		bool GetLabelData(std::vector<uint8_t>& data);
		bool GetLabelData(void* data);
		bool Close();
};


}
}
