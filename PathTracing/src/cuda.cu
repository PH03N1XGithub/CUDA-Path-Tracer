#include "cuda.h"
#include "RayTraceLib.cu"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math_functions.h>
#include <vector_types.h> 

struct CudaCube {
    float3 Position;     // Center of the cube
    float3 HalfSize;     // Half-extent in each direction (axis-aligned box)
    int MaterialIndex;
};
__device__ void my_swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ bool IntersectCube(const CudaRay& ray, const CudaCube& cube, float& tHit) {
    float3 min = cube.Position - cube.HalfSize;
    float3 max = cube.Position + cube.HalfSize;

    float tMin = (min.x - ray.Origin.x) / ray.Direction.x;
    float tMax = (max.x - ray.Origin.x) / ray.Direction.x;

    if (tMin > tMax) my_swap(tMin, tMax);

    float tyMin = (min.y - ray.Origin.y) / ray.Direction.y;
    float tyMax = (max.y - ray.Origin.y) / ray.Direction.y;

    if (tyMin > tyMax) my_swap(tyMin, tyMax);

    if ((tMin > tyMax) || (tyMin > tMax)) return false;

    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;

    float tzMin = (min.z - ray.Origin.z) / ray.Direction.z;
    float tzMax = (max.z - ray.Origin.z) / ray.Direction.z;

    if (tzMin > tzMax) my_swap(tzMin, tzMax);

    if ((tMin > tzMax) || (tzMin > tMax)) return false;

    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;

    tHit = tMin;
    return tHit > 0;
}


// ----------------------------------------------------------------------------------
// Hit computations
// ----------------------------------------------------------------------------------
__device__ CudaHitPayload ComputeHit(const CudaRay& ray, float hitDistance, int objectIndex,const CudaSphere* spheres)
{
    CudaHitPayload payload;
    payload.HitDistance = hitDistance;
    payload.ObjectIndex = objectIndex;

    const CudaSphere& closestSphere = spheres[objectIndex];

    const float3 origin = ray.Origin - closestSphere.Position;
    payload.WorldPosition = origin + ray.Direction * hitDistance;
    payload.WorldNormal = normalize(payload.WorldPosition);

    payload.WorldPosition += closestSphere.Position;

    return payload;
}

__device__ CudaHitPayload Miss(const CudaRay& ray)
{
    CudaHitPayload payload;
    payload.HitDistance = -1.0f;
    return payload;
}




// ----------------------------------------------------------------------------------
// Ray-sphere intersection
// ----------------------------------------------------------------------------------
__device__ CudaHitPayload TraceRay(const CudaRay& ray, const CudaSphere* spheres, int numSpheres,const CudaCube* cubes, int numCubes) {
    CudaHitPayload payload;
    payload.HitDistance = -1.0f;
    payload.ObjectIndex = -1;

    for (int i = 0; i < numSpheres; i++) {
        const CudaSphere& s = spheres[i];
        float3 oc = ray.Origin - s.Position;
        float b = 2.0f * dot3(ray.Direction, oc);
        float c = dot3(oc, oc) - s.Radius * s.Radius;
        float disc = b * b - 4.0f * c;
        if (disc <= 0.0f) continue;

        float sqrtD = sqrtf(disc);
        float t1 = (-b - sqrtD) * 0.5f;
        float t2 = (-b + sqrtD) * 0.5f;
        float t = (t1 > 0.0f ? t1 : (t2 > 0.0f ? t2 : -1.0f));
        if (t > 0.0f && (payload.HitDistance < 0.0f || t < payload.HitDistance)) {
            payload.HitDistance = t;
            payload.ObjectIndex = i;
        }
    }
    // Cube hits
    for (int i = 0; i < numCubes; i++) {
        float t;
        if (IntersectCube(ray, cubes[i], t)) {
            if (t > 0.0f && (payload.HitDistance < 0.0f || t < payload.HitDistance)) {
                payload.HitDistance = t;
                payload.ObjectIndex = 10000 + i; // Offset index so you know it’s a cube
            }
        }
    }

    if (payload.ObjectIndex < 0)
        return Miss(ray);

    // Differentiate handling for spheres/cubes
    if (payload.ObjectIndex < 10000) {
        return ComputeHit(ray, payload.HitDistance, payload.ObjectIndex, spheres);
    } else {
        int cubeIdx = payload.ObjectIndex - 10000;
        const CudaCube& cube = *cubes;
        CudaHitPayload hit;
        hit.HitDistance = payload.HitDistance;
        hit.ObjectIndex = payload.ObjectIndex;
        hit.WorldPosition = ray.Origin + ray.Direction * hit.HitDistance;

        // Compute normal based on which face was hit (not exact but fast)
        float3 localPos = hit.WorldPosition - cube.Position;
        localPos.x = fabs(localPos.x);
        localPos.y = fabs(localPos.y);
        localPos.z = fabs(localPos.z);
        float3 n;
        float3 d = localPos - cube.HalfSize;
        if (d.x > d.y && d.x > d.z)
            n = make_float3(signbit(localPos.x) ? -1.0f : 1.0f, 0.0f, 0.0f);
        else if (d.y > d.z)
            n = make_float3(0.0f, signbit(localPos.y) ? -1.0f : 1.0f, 0.0f);
        else
            n = make_float3(0.0f, 0.0f, signbit(localPos.z) ? -1.0f : 1.0f);

        hit.WorldNormal = n;
        return hit;
    }
}

// ----------------------------------------------------------------------------------
// Ray-tracing kernel
// ----------------------------------------------------------------------------------
extern "C" __global__ void RayTraceKernel(
    const CudaSphere* spheres,
    const CudaMaterial* materials,
    int numSpheres,
    uint32_t* image,
    float3* accumulation,
    int width,
    int height,
    CudaCamera camera,
    int frameIndex,
    bool bSkyBox,
    int maxBounce,
    int SPP,
    const CudaCube* cubes
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    int samplesPerPixel = SPP;
    int maxBounces = maxBounce;
    /*if (frameIndex == 1)
    {
        samplesPerPixel = 1;
    }*/

    float3 colorSum = make_float3(0.0f,0.0f,0.0f);



    for (int s = 0; s < samplesPerPixel; s++) {
        uint32_t seed = idx * 9781 + frameIndex * 684 + s * 917;

        float jitterX = RandomFloat(seed) - 0.5f;
        float jitterY = RandomFloat(seed) - 0.5f;

        float u = (static_cast<float>(x) + 0.5f + jitterX) / width;
        float v = (static_cast<float>(y) + 0.5f + jitterY) / height;

        float aspect = static_cast<float>(width) / static_cast<float>(height);
        float scale = tanf(camera.FovY * -0.5f);

        float px = (2.0f * u - 1.0f) * aspect * scale;
        float py = (1.0f - 2.0f * v) * scale;

        float3 forward = normalize(camera.ForwardDirection);
        float3 right = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), forward));
        float3 up = normalize(cross(forward, right));

        float3 direction = normalize(px * right + py * up + forward);


        float3 randomInLens =  RandomInUnitSphere(seed) * camera.Aperture * 0.5f;
        float3 offset = right * randomInLens.x + up * randomInLens.y;

        // Compute the point on the focus plane
        float3 target = camera.Position + direction * camera.FocusDistance;

        CudaRay ray;
        ray.Origin = camera.Position + offset;
        ray.Direction = normalize(target - ray.Origin);

        float3 light = make_float3(0.0f,0.0f,0.0f);
        float3 contrib = make_float3(1.0f,1.0f,1.0f);

        for (int bounce = 0; bounce < maxBounces; bounce++) {
            CudaHitPayload hit = TraceRay(ray, spheres, numSpheres,cubes,1);
            if (hit.HitDistance < 0.0f) {
                light += contrib * make_float3(0.8f, 0.8f, 0.9f) * bSkyBox; // Skybox
                break;
            }
            
            

            const CudaSphere& sphere = spheres[hit.ObjectIndex];
            const CudaMaterial& mat = materials[sphere.MaterialIndex];

            light += contrib * mat.Emission;

            float3 N = hit.WorldNormal;
            float3 R = RandomInUnitSphere(seed);
            float3 diffuseDir = normalize(N + R);
            float3 perturbedN = normalize(N + mat.Roughness * R);
            float3 specularDir = reflect(ray.Direction, perturbedN);

            ray.Direction = normalize((1.0f - mat.Metallic) * diffuseDir + specularDir);
            ray.Origin = hit.WorldPosition + ray.Direction * 0.0001f;

            float3 F0 = lerp(make_float3(0.04f,0.04f,0.04f), mat.Albedo, mat.Metallic);
            float3 diffuseColor = (1.0f - mat.Metallic) * mat.Albedo;
            float3 specularColor = F0;

            contrib *= (diffuseColor + specularColor);
        }

        colorSum += light;
    }

    float3 frameColor = colorSum / static_cast<float>(samplesPerPixel);

    // Blend with previous accumulation
    float3 prev = accumulation[idx];
    float alpha = 1.0f / static_cast<float>(frameIndex); // Temporal blend factor
    float3 newAccum = lerp(prev, frameColor, alpha);
    accumulation[idx] = newAccum;

    image[idx] = PackRGBA(
        static_cast<uint8_t>(fminf(newAccum.x, 1.0f) * 255.0f),
        static_cast<uint8_t>(fminf(newAccum.y, 1.0f) * 255.0f),
        static_cast<uint8_t>(fminf(newAccum.z, 1.0f) * 255.0f),
        255
    );
}



// ----------------------------------------------------------------------------------
// Host Launcher (Ray Trace)
// ----------------------------------------------------------------------------------
static int frameIndex = 1;
void RunCudaRayTrace(uint32_t* hostImage, int width, int height, const CudaSphere* hostSpheres,
    const CudaMaterial* hostMaterials, int numSpheres, const CudaCamera& camera, bool bAccumulate ,bool bSkyBox,int maxBounces, int samplesPerPixel) {
    static uint32_t* devImage = nullptr;
    static float3* devAccumulation = nullptr;
    static int lastWidth = 0, lastHeight = 0;
    int devMaxBounce = maxBounces, devSamplesPerPixel = samplesPerPixel;
    CudaSphere* devSpheres = nullptr;
    CudaMaterial* devMaterials = nullptr;


    size_t imgBytes = static_cast<size_t>(width) * height * sizeof(uint32_t);
    size_t accumBytes = static_cast<size_t>(width) * height * sizeof(float3);
    size_t sphBytes = static_cast<size_t>(numSpheres) * sizeof(CudaSphere);
    size_t matBytes = static_cast<size_t>(numSpheres) * sizeof(CudaMaterial);

    if (width != lastWidth || height != lastHeight) {
        std::cout << "Resize\n";
        lastWidth = width;
        lastHeight = height;
        CudaResetFrameIndex();
    }
    cudaMalloc(&devImage, imgBytes);
    if (!devAccumulation) {
        cudaMalloc(&devAccumulation, accumBytes);
        std::cout << "Resolution changed, freeing and resetting accumulation buffer\n";
    }
    
    cudaMalloc(&devSpheres, sphBytes);
    cudaMalloc(&devMaterials, matBytes);
    

    cudaMemcpy(devSpheres, hostSpheres, sphBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devMaterials, hostMaterials, matBytes, cudaMemcpyHostToDevice);

    CudaCamera CudaCamera;
    CudaCamera.Position = camera.Position;
    CudaCamera.ForwardDirection = normalize(camera.ForwardDirection);

    const float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
    CudaCamera.Right = normalize(cross(worldUp, CudaCamera.ForwardDirection));
    CudaCamera.Up = normalize(cross(CudaCamera.Right, CudaCamera.ForwardDirection));
    CudaCamera.FovY = camera.FovY * 3.14159265f / 180.0f;
    CudaCamera.Aperture = camera.Aperture;
    CudaCamera.FocusDistance = camera.FocusDistance;


    CudaCube* hcubes = new CudaCube;
    hcubes->Position = make_float3(100.0f, 0.0f, 100.0f);
    hcubes->HalfSize = make_float3(1000.0f, 0.0f, 1000.0f);
    
    size_t cubeBytes = static_cast<size_t>(1) * sizeof(CudaCube);
    static CudaCube* devCubes = nullptr;
    cudaMalloc(&devCubes, cubeBytes);
    cudaMemcpy(devCubes, hcubes, cubeBytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    RayTraceKernel<<<blocks, threads>>>(
        devSpheres,
        devMaterials,
        numSpheres,
        devImage,
        devAccumulation,
        width,
        height,
        CudaCamera,
        frameIndex,
        bSkyBox,
        maxBounces,
        samplesPerPixel,
        devCubes
    );
    

    if (bAccumulate)
        frameIndex++;
    else
        frameIndex = 1;

    cudaDeviceSynchronize();
    cudaMemcpy(hostImage, devImage, imgBytes, cudaMemcpyDeviceToHost);

    cudaFree(devImage);
    cudaFree(devSpheres);
    cudaFree(devMaterials);
    //cudaFree(devAccumulation);
}



// ----------------------------------------------------------------------------------
// Simple UV gradient kernel
// ----------------------------------------------------------------------------------
extern "C" __global__
void UVGradientKernel(uint32_t* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t r = uint8_t(float(x) / width * 255);
    uint8_t g = uint8_t(float(y) / height * 255);
    image[y * width + x] = PackRGBA(r, g, 0, 255);
}

// ----------------------------------------------------------------------------------
// Host Launcher (UV Gradient)
// ----------------------------------------------------------------------------------

void RunCudaGradient(uint32_t* hostImage, int width, int height) {
    size_t bytes = size_t(width) * height * sizeof(uint32_t);
    uint32_t* devImage = nullptr;
    cudaMalloc(&devImage, bytes);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    UVGradientKernel<<<blocks, threads>>>(devImage, width, height);

    cudaDeviceSynchronize();
    cudaMemcpy(hostImage, devImage, bytes, cudaMemcpyDeviceToHost);
    cudaFree(devImage);
}

void CudaResetFrameIndex() { frameIndex = 1; }


