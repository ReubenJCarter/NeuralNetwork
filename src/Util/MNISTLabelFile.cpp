#include "MNISTLabelFile.h"

#include "FlipBytes.h"

#include <stdint.h>
#include <iostream>
#include <fstream>


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
}

bool MNISTLabelFile::Open(const char* fileName)
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
	
	isOpen = true; 
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
	
	fileS.seekg(sizeof(uint32_t) * 2 + labelIndex, std::ios::beg);
	fileS.read((char*)&data, sizeof(uint8_t));
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
	if(isOpen)
	{
		fileS.close();
		isOpen = false;
		return true; 
	}
	return false;
}


}
}