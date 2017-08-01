#include "LoadImage.h"



namespace NN
{	
namespace Util
{
	

void LoadImage(void* data, int& width, int& height, const char* fn)
{
	ilInit();
	
	//make new deil image and load from file name
	ILuint ilIm;
	ilGenImages(1, &ilIm);
	ilBindImage(ilIm);
	bool loadedImageOK;
	loadedImageOK = ilLoadImage(fn);
	if(loadedImageOK)
	{
		//get image info
		int widthIL = ilGetInteger(IL_IMAGE_WIDTH);
        int heightIL = ilGetInteger(IL_IMAGE_HEIGHT);
		
		//convert data to B4 type and format
		ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE); 
		Deallocate();
		
		//copy data
		uint8_t* dataPtr = ilGetData();
		if(dataPtr == NULL)
		{
			loadedImageOK = false;
			std::cerr << "Image:" << "No data for image:" << fn << std::endl;
			return false;
		}
		else
		{
			width = widthIL;
			height = heightIL;
			data = new uint8_t[width * height];
			for(unsigned int i = 0; i < width * height * 4; i++)
			{
				((uint8_t*)data)[i] = dataPtr[i];
			}
		}
	}
	else
	{
		std::cerr << "Image:" << "Could not load image file:" << fn << std::endl;
		return false; 
	}
	ilBindImage(0);
	ilDeleteImages(1, &ilIm);
	ilBindImage(ilIm);
	return true;
}
	
	
}
}