#pragma once 

#include <iostream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <iostream>

namespace CLHelper
{	


class PlatformInfo;
	
class DeviceInfo
{
public:
	int platformIndex;
	cl_platform_id platformId;
	cl_device_id deviceId;
	std::string name;
	bool avalible;
	std::string type;
	
	DeviceInfo(cl_device_id did, int pinx, cl_platform_id pid)
	{
		platformId = pid;
		platformIndex = pinx;
		deviceId = did;
		Name(name);
		Avalible(avalible);
		DeviceType(type);
	}
	
	bool Name(std::string& nameStr)
	{
		const unsigned int NAME_BUFFER_SIZE = 10000;
		char nameBuffer[NAME_BUFFER_SIZE];
		size_t returnBufferSize;
		cl_int deviceInfoErr = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, NAME_BUFFER_SIZE, nameBuffer, &returnBufferSize);
		if(deviceInfoErr != CL_SUCCESS)
		{
			std::cerr << "clGetDeviceInfo CL_DEVICE_NAME error" << std::endl;
			nameStr = std::string();
			return false;
		}
		nameStr = std::string(nameBuffer, returnBufferSize);
		return true;
	}
	
	bool Avalible(bool& avalibleBool)
	{
		cl_bool avail;
		cl_int deviceInfoErr = clGetDeviceInfo(deviceId, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &avail, NULL);
		if(deviceInfoErr != CL_SUCCESS)
		{
			std::cerr << "clGetDeviceInfo CL_DEVICE_AVAILABLE error" << std::endl;
			avalibleBool = false;
			return false;
		}
		avalibleBool = (bool)avail;
		return true;
	}
	
	bool DeviceType(std::string& typeStr)
	{
		cl_device_type type;
		cl_int deviceInfoErr = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
		if(deviceInfoErr != CL_SUCCESS)
		{
			std::cerr << "clGetDeviceInfo CL_DEVICE_TYPE error" << std::endl;
			typeStr = std::string();
			return false;
		}
		if(type == CL_DEVICE_TYPE_CPU)
			typeStr = "CPU";
		else if(type == CL_DEVICE_TYPE_GPU)
			typeStr = "GPU";
		else if(type == CL_DEVICE_TYPE_ACCELERATOR)
			typeStr = "ACCELERATOR";
		else if(type == CL_DEVICE_TYPE_DEFAULT)
			typeStr = "DEFAULT";
		else 
			typeStr = "??";
		
		return true;
	}
};


class PlatformInfo
{
public:
	cl_platform_id platformId;
	std::string name;
	std::vector<DeviceInfo> deviceInfo;
	
	PlatformInfo(cl_platform_id pid)
	{
		platformId = pid;
		Name(name);
	}
	
	bool Name(std::string& nameStr)
	{
		const unsigned int NAME_BUFFER_SIZE = 10000;
		char nameBuffer[NAME_BUFFER_SIZE];
		size_t returnBufferSize;
		cl_int deviceInfoErr = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, NAME_BUFFER_SIZE, nameBuffer, &returnBufferSize);
		if(deviceInfoErr != CL_SUCCESS)
		{
			std::cerr << "clGetPlatformInfo CL_DEVICE_NAME error" << std::endl;
			nameStr = std::string();
			return false;
		}
		nameStr = std::string(nameBuffer, returnBufferSize);
		return true;
	}
};


class CLPlatforms
{
public:
	std::vector<PlatformInfo> platformInfo;

	CLPlatforms()
	{
		GetPlatformAndDeviceInfo();
	}
	
	bool GetPlatformAndDeviceInfo()
	{
		//Get the avalible platforms
		unsigned int platformNumber;
		cl_int getPlatformErr = clGetPlatformIDs(0, NULL, &platformNumber);
		if(getPlatformErr != CL_SUCCESS)
		{
			std::cerr << "getPlatformErr" << std::endl;
			return false;
		}
		cl_platform_id* platformIds = new cl_platform_id[platformNumber];
		getPlatformErr = clGetPlatformIDs(platformNumber, platformIds, NULL);
		if(getPlatformErr != CL_SUCCESS)
		{
			std::cerr << "getPlatformErr" << std::endl;
			return false;
		}
		
		//Get each availible device on each platdform
		platformInfo.clear();
		for(unsigned int i = 0; i < platformNumber; i++)
		{
			//get the platforms
			unsigned int deviceNumber;
			cl_int getDeviceErr = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceNumber);
			if(getDeviceErr != CL_SUCCESS)
			{
				std::cerr << "clGetDeviceIDs" << std::endl;
				return false;
			}
			cl_device_id* deviceIds = new cl_device_id[deviceNumber];
			getDeviceErr = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceNumber, deviceIds, NULL);
			if(getDeviceErr != CL_SUCCESS)
			{
				std::cerr << "clGetDeviceIDs" << std::endl;
				return false;
			}
			
			//Build vector of platforms with devices
			PlatformInfo platInfo(platformIds[i]);
			platformInfo.push_back(platInfo);
			for(unsigned j = 0; j < deviceNumber; j++)
			{
				DeviceInfo devInfo(deviceIds[j], i, platformIds[i]);
				
				//Add to device info
				platformInfo[i].deviceInfo.push_back(devInfo);
			}
			
			
			if(deviceNumber > 0) delete[] deviceIds;
		}
		
		if(platformNumber > 0) delete[] platformIds;
		return true; 
	}
	
	DeviceInfo* GetFirstGPUDeviceInfo()
	{
		for(int i = 0; i < platformInfo.size(); i++)
		{
			for(int j = 0; j < platformInfo[i].deviceInfo.size(); j++)
			{
				if(platformInfo[i].deviceInfo[j].type == "GPU")
				{
					return &(platformInfo[i].deviceInfo[j]);
				}
			}
		}
		return NULL;
	}
};


}