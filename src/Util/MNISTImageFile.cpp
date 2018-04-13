#include "MNISTImageFile.h"

#include "FlipBytes.h"


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
	dataPreloaded = false;
}

bool MNISTImageFile::Open(const char* fileName, bool preload)
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
	
	dataPreloaded = preload;
	
	isOpen = true; 
	
	preloadedData.clear();
	if(dataPreloaded)
	{
		preloadedData.resize(width * height * imageNumber);
		fileS.read((char*)&preloadedData[0], width * height * imageNumber);
		fileS.close();
	}
	
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
	
	data.resize(width * height);
	if(!dataPreloaded)
	{
		fileS.seekg(sizeof(uint32_t) * 4 + imageIndex * width * height, std::ios::beg);
		fileS.read((char*)&data[0], width * height);
	}
	else
	{
		for(int i = 0; i < width * height; i++)
			data[i] = preloadedData[width * height * imageIndex + i];
	}
	return true;
}

bool MNISTImageFile::GetImageData(int imageIndex, void* data)
{
	if(!isOpen)
		return false;	
	
	if(!dataPreloaded)
	{
		fileS.seekg(sizeof(uint32_t) * 4 + imageIndex * width * height, std::ios::beg);
		fileS.read((char*)data, width * height);
	}
	else
	{
		for(int i = 0; i < width * height; i++)
			((uint8_t*)data)[i] = preloadedData[width * height * imageIndex + i];
	}
	return true;
}

bool MNISTImageFile::GetImageDataAsFloat(int imageIndex, void* dataFloat)
{
	if(!isOpen)
		return false;	
	
	if(!dataPreloaded)
	{
		std::vector<uint8_t> data(width * height); 
		fileS.seekg(sizeof(uint32_t) * 4 + imageIndex * width * height, std::ios::beg);
		fileS.read((char*)&data[0], width * height);
		for(int i = 0; i < width * height; i++)
			((float*)dataFloat)[i] = ((float)data[i]) / 255.0;
	}
	else
	{
		for(int i = 0; i < width * height; i++)
			((float*)dataFloat)[i] = ((float)preloadedData[width * height * imageIndex + i]) / 255.0;
	}
	
	return true;
}

bool MNISTImageFile::Close()
{
	if(isOpen && !dataPreloaded)
	{
		fileS.close();
		isOpen = false;
		return true; 
	}
	return false;
}


}
}