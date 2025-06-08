#pragma once
#include "Walnut/Image.h"
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"
#include "CudaPathTrace.h"
#include <memory>
#include <glm/glm.hpp>

#ifdef WL_RELEASE_GPU
#define WL_RENDERER_PROSES_UNIT "GPU"
#else
#ifdef WL_RELEASE
#define WL_RENDERER_PROSES_UNIT "CPU Render"
#endif
#endif

class Renderer
{
public:
	struct Settings
	{
		bool Accumulate = true;
		bool SkyBox = true;
		int maxBounces = 3, samplesPerPixel = 1;
	};
public:
	Renderer() = default;

	void OnResize(uint32_t width, uint32_t height);
	void Render(const Scene& scene, const Camera& camera);


	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }

	Settings& GetSettings() { return m_Settings; }
	void ResetFrameIndex();


public:
	struct HitPayload
	{
		float HitDistance;
		glm::vec3 WorldPosition;
		glm::vec3 WorldNormal;

		int ObjectIndex;
	};

	std::vector<HitPayload> TraceRaysGPU(const std::vector<Ray>& rays);

	glm::vec4 PerPixel(uint32_t x, uint32_t y); // RayGen

	HitPayload TraceRay(const Ray& ray);
	HitPayload ClosestHit(const Ray& ray, float hitDistance, int objectIndex);
	HitPayload Miss(const Ray& ray);
private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	Settings m_Settings;

	std::vector<uint32_t> m_ImageHorizontalIter, m_ImageVerticalIter;

	const Scene* m_ActiveScene = nullptr;
	const Camera* m_ActiveCamera = nullptr;

	uint32_t* m_ImageData = nullptr;
	glm::vec4* m_AccumulationData = nullptr;
	uint32_t m_FrameIndex = 1;
public:
	
	float LastSetDataTime = 0.0f;
	float LastRayTraceTime = 0.0f;
	
	std::string ProsesUnit = WL_RENDERER_PROSES_UNIT;
};



