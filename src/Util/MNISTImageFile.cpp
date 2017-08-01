#include "MNISTImageFile.h"

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
 * MNISTImageFile
 *
 */


MNISTImageFile::MNISTImageFile()
{
	isOpen = false;
}

bool MNISTImageFile::Open(const char* fileName)
{
	if(isOpen)
		fileS.close();
	isOpen = false;
	fileS.open(fileName, std::ios::binary);
	if(!fileS.is_open())
	{
		std::cerr << "MNISTImageFile: open fail" << std::endl;
		return false;
	}
		
	fileS.read((char*)(&magicNumber), sizeof(uint32_t));
	FilpBytes(&magicNumber, sizeof(uint32_t));
	if(magicNumber != 2051)
	{
		std::cerr << "MNISTImageFile: magic number fail:" << magicNumber << std::endl;
		return false;
	}
		
	fileS.read((char*)(&imageNumber), sizeof(uint32_t));
	FilpBytes(&imageNumber, sizeof(uint32_t));
	fileS.read((char*)(&height), sizeof(uint32_t));
	FilpBytes(&height, sizeof(uint32_t));
	fileS.read((char*)(&width), sizeof(uint32_t));
	FilpBytes(&width, sizeof(uint32_t));
	
	isOpen = true; 
	return true;
}

int MNISTImageFile::GetImageNumber()
{
	return imageNumber;
}

int MNISTImageFile::GetWidth()
{
	return width;
}

int MNISTImageFile::GetHeight()
{
	return height;
}

bool MNISTImageFile::GetImageData(int imageIndex, std::vector<uint8_t>& data)
{
	if(!isOpen)
		return false;	
	
	fileS.seekg(sizeof(uint32_t) * 4 + imageIndex * width * height, std::ios::beg);
	data.resize(width * height);
	fileS.read((char*)&data[0], width * height);
	return true;
}

bool MNISTImageFile::GetImageData(int imageIndex, void* data)
{
	if(!isOpen)
		return false;	
	
	fileS.seekg(sizeof(uint32_t) * 4 + imageIndex * width * height, std::ios::beg);
	fileS.read((char*)data, width * height);
	return true;
}

bool MNISTImageFile::Close()
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