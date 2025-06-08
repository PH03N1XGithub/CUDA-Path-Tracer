#pragma once
#include "CudaStructs.h"

void RunCudaRayTrace(uint32_t* hostImage, uint32_t width, uint32_t height, const CudaSphere* hostSpheres, const CudaMaterial* hostMaterials, int numSpheres,
	const CudaCamera& camera, bool bAccumulate,bool bSkyBox,int maxBounces, int samplesPerPixel);

void RunCudaGradient(uint32_t* hostImage, int width, int height);

void CudaResetFrameIndex();