#pragma once
#include "CudaStructs.h"

// ----------------------------------------------------------------------------------
// Operator overloads
// ----------------------------------------------------------------------------------

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ __host__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

__host__ __device__ inline float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __host__ inline float4& operator/=(float4& v, float scalar) {
    float inv = 1.0f / scalar;
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    v.w *= inv;
    return v;
}


// ----------------------------------------------------------------------------------
// Math helpers
// ----------------------------------------------------------------------------------

__host__ __device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float clamp(float val, float minVal, float maxVal)
{
    return fminf(fmaxf(val, minVal), maxVal);
}

__device__ inline float3 clamp(float3 v, float minVal, float maxVal)
{
    return make_float3(
        clamp(v.x, minVal, maxVal),
        clamp(v.y, minVal, maxVal),
        clamp(v.z, minVal, maxVal)
    );
}

__host__ __device__ inline float3 normalize(const float3& v, float epsilon = 1e-8f) {
    float lenSq = dot3(v, v);
    if (lenSq < epsilon)
        return make_float3(0.0f, 0.0f, 0.0f);
    return v * rsqrtf(lenSq);
}

__host__ __device__ inline uint32_t PackRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (static_cast<uint32_t>(a) << 24)
         | (static_cast<uint32_t>(b) << 16)
         | (static_cast<uint32_t>(g) << 8)
         | static_cast<uint32_t>(r);
}

__device__ inline float RandomFloat(uint32_t& seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (seed * 16807 % 2147483647) / 2147483647.0f;
}

__device__ inline float3 RandomInUnitSphere(uint32_t& seed) {
    float3 p;
    do {
        p = make_float3(RandomFloat(seed), RandomFloat(seed), RandomFloat(seed));  
        
        p.x = p.x * 2.0f - 1.0f; 
        p.y = p.y * 2.0f - 1.0f;
        p.z = p.z * 2.0f - 1.0f;
    } while (false); 
    return p;
}

__host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

__host__ __device__ inline float3 reflect(const float3& I, const float3& N) {
    return I - 2.0f * dot3(N, I) * N;
}

__device__ inline float4 mul(const float4x4& m, const float4& v) {
    return make_float4(
        m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
        m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
        m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
        m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w
    );
}


__device__ inline void swap(float& a, float& b) noexcept
{
    const float temp = a;
    a = b;
    b = temp;
}