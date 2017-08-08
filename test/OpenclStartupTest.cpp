#include "Util/CommonHeaders.h"
#include "OpenCLHelper/CLPlatforms.h"


using namespace CLHelper;


int main(int argc, char* argv[])
{
	
	CLPlatforms platforms;
	for(int i = 0; i < platforms.platformInfo.size(); i++)
	{
		std::cout << "platform:" << platforms.platformInfo[i].name << std::endl;
		for(int j = 0; j < platforms.platformInfo[i].deviceInfo.size(); j++)
		{
			std::cout << std::endl; 
			std::cout << "device:" << platforms.platformInfo[i].deviceInfo[j].name << std::endl;
			std::cout << "avalible:" << platforms.platformInfo[i].deviceInfo[j].avalible << std::endl;
			std::cout << "type:" << platforms.platformInfo[i].deviceInfo[j].type << std::endl;
		}
	}
	
	std::cout << std::endl; 
	std::cout << "First GPU Device" << std::endl;
	
	DeviceInfo* dinfo = platforms.GetFirstGPUDeviceInfo();
	
	std::cout << "device:" << dinfo->name << std::endl;
	std::cout << "avalible:" << dinfo->avalible << std::endl;
	std::cout << "type:" << dinfo->type << std::endl;
	
	return 1;
}