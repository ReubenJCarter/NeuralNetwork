#include "SaveImage.h"

#include <iostream>
#include <stdint.h>
#include <IL/il.h>


namespace NN
{	
namespace Util
{
	

bool SaveImage(void* data, int& width, int& height, const char* fn)
{
	ilInit();
	
	uint8_t* cdata = new uint8_t[width * height];
	
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			cdata[(height - 1 - i) * width + j] = data[i * width + j];
		}
	}
	
	ILboolean success;
	if(width > 0 && height > 0)
	{
		ILuint ilimage;
		ilGenImages(1, &ilimage);
		ilBindImage(ilimage);
				
		success = ilTexImage(width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, cdata);
		if(success!=true)
		{
			std::cerr << "IMAGE:failed to copy image data for saving" << std::endl;
			return false;
		}
		else
		{
			ilEnable(IL_FILE_OVERWRITE);
			ilSaveImage(filename);
			ilDisable(IL_FILE_OVERWRITE);
			ilDeleteImages(1, &ilimage);
		}
	}
	else
	{
		std::cerr << "IMAGE:no data in image" << std::endl;
		success = false;
		return false;
	}
	
	delete[] cdata;
	return true;
}


}
}