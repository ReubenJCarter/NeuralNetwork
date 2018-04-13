#include "MNISTLabelFile.h"

#include "FlipBytes.h"


namespace NN
{	
namespace Util
{


/*
 *
 * MNISTLabelFile
 *
 */


MNISTLabelFile::MNISTLabelFile()
{
	isOpen = false;
	dataPreloaded = false;
}

bool MNISTLabelFile::Open(const char* fileName, bool preload)
{
	if(isOpen)
		fileS.close();
	isOpen = false;
	fileS.open(fileName, std::ios::binary);
	if(!fileS.is_open())
	{
		std::cerr << "MNISTLabelFile: open fail" << std::endl;
		return false;
	}
		
	fileS.read((char*)(&magicNumber), sizeof(uint32_t));
	FilpBytes(&magicNumber, sizeof(uint32_t));
	if(magicNumber != 2049)
	{
		std::cerr << "MNISTLabelFile: magic number fail:" << magicNumber << std::endl;
		return false;
	}
		
	fileS.read((char*)(&labelNumber), sizeof(uint32_t));
	FilpBytes(&labelNumber, sizeof(uint32_t));
	
	dataPreloaded = preload;
	
	isOpen = true; 
	
	preloadedData.clear();
	if(dataPreloaded)
	{
		preloadedData.resize(labelNumber);
		fileS.read((char*)&preloadedData[0], labelNumber * sizeof(uint8_t));
		fileS.close();
	}
	
	return true;
}

int MNISTLabelFile::GetLabelNumber()
{
	return labelNumber;
}

bool MNISTLabelFile::GetLabelData(int labelIndex, uint8_t& data)
{
	if(!isOpen)
		return false;
	
	if(!dataPreloaded)
	{
		fileS.seekg(sizeof(uint32_t) * 2 + labelIndex, std::ios::beg);
		fileS.read((char*)&data, sizeof(uint8_t));
	}
	else
	{
		data = preloadedData[labelIndex];
	}
	return true;
}

bool MNISTLabelFile::GetLabelData(std::vector<uint8_t>& data)
{
	if(!isOpen)
		return false;
		
	fileS.seekg(sizeof(uint32_t) * 2, std::ios::beg);
	data.resize(labelNumber);
	fileS.read((char*)&data[0], labelNumber * sizeof(uint8_t));
	return true;
}

bool MNISTLabelFile::GetLabelData(void* data)
{
	if(!isOpen)
		return false;
		
	fileS.seekg(sizeof(uint32_t) * 2, std::ios::beg);
	fileS.read((char*)data, labelNumber * sizeof(uint8_t));
	return true;
}

bool MNISTLabelFile::Close()
{
	if(isOpen && !dataPreloaded)
	{
		
		fileS.close();
		isOpen = false;
		return true; 
	}
	else if(isOpen && dataPreloaded)
	{
		isOpen = false;
		dataPreloaded = false;
		preloadedData.clear();
		return true;
	}
	return false;
}


}
}