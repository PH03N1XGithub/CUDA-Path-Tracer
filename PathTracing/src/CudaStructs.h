#pragma once
#include <vector_types.h>

struct CudaRay
{
	float3 Origin;   // Ray origin
	float3 Direction; // Ray direction
};

struct CudaMaterial
{
	float3 Albedo;   // Surface color (RGB)
	float3 Emission; // Light emission from the material
	float Roughness;    // Surface roughness (for specular reflection)
	float Metallic;     // Material's metallicity (0 for dielectrics, 1 for metals)
};

struct CudaSphere
{
	float3 Position; // Sphere position in world space
	float Radius;       // Sphere radius
	int MaterialIndex;  // Index of the material for the sphere
};

struct CudaHitPayload
{
	float HitDistance;     // Distance at which the ray hit something, negative if missed
	int ObjectIndex;       // Index of the object that was hit (sphere index)
	float3 WorldPosition; // World position of the hit point
	float3 WorldNormal;   // Normal at the hit point
};
struct float4x4 {
	float m[4][4];

	__device__ __host__
	float* operator[](int row) {
		return m[row];
	}

	__device__ __host__
	const float* operator[](int row) const {
		return m[row];
	}
};



struct CudaCamera {
	float3 Position;
	float3 ForwardDirection;
	float3 Up;
	float3 Right;
	float FovY;
	float AspectRatio;
	float NearClip;
	float FarClip;
	float Aperture = 0.1f;         // Controls blur intensity
	float FocusDistance = 10.0f;   // Distance to the focus plane


	// NEW: Matrices
	float4x4 ViewMatrix;
	float4x4 InverseViewMatrix;
	float4x4 ProjectionMatrix;
	float4x4 InverseProjectionMatrix;

};
struct CudaCube {
	float3 Position;     // Center of the cube
	float3 HalfSize;     // Half-extent in each direction (axis-aligned box)
	int MaterialIndex;
};
