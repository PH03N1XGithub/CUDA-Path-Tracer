#include "Renderer.h"
#include <execution>
#include <fstream>
#include <vector_functions.h>
#include "Walnut/Random.h"
#include "Walnut/Application.h"
#include "Walnut/Timer.h"
#include "cuda.h"

namespace Utils {

	static uint32_t ConvertToRGBA(const glm::vec4& color)
	{
		uint8_t R = (uint8_t)(color.r * 255.0f);
		uint8_t G = (uint8_t)(color.g * 255.0f);
		uint8_t B = (uint8_t)(color.b * 255.0f);
		uint8_t A = (uint8_t)(color.a * 255.0f);

		uint32_t result = (A << 24) | (B << 16) | (G << 8) | R;
		return result;
	}

}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_FinalImage)
	{
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
			return;
		m_FinalImage->Resize(width, height);
	}
	else
	{
		m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}

	delete[] m_ImageData;
	m_ImageData = new uint32_t[width * height];

	delete[] m_AccumulationData;
	m_AccumulationData = new glm::vec4[width * height];

	m_ImageHorizontalIter.resize(width);
	m_ImageVerticalIter.resize(height);
	
	for (uint32_t i = 0; i < width; i++)
		m_ImageHorizontalIter[i] = i;
	for (uint32_t i = 0; i < height; i++)
		m_ImageVerticalIter[i] = i;
}



void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_ActiveScene = &scene;
	m_ActiveCamera = &camera;

#if WL_RELEASE_GPU
	Walnut::Timer timer;

	const size_t numSpheres = m_ActiveScene->Spheres.size();
	const size_t numMaterials = m_ActiveScene->Materials.size();

	std::vector<CudaSphere> c_spheres(numSpheres);
	std::vector<CudaMaterial> c_materials(numMaterials);
	
	for (size_t i = 0; i < numSpheres; i++) {
		const Sphere& s = m_ActiveScene->Spheres[i];
		const Material& m = m_ActiveScene->Materials[s.MaterialIndex];

		c_spheres[i] = s;
		c_materials[s.MaterialIndex] = m;
	}
	
	CudaCamera cudaCamera = Camera::GetGetCudaCamera(*m_ActiveCamera, m_FinalImage->GetWidth(), m_FinalImage->GetHeight());
	
	RunCudaRayTrace(
		m_ImageData,
		m_FinalImage->GetWidth(),
		m_FinalImage->GetHeight(),
		c_spheres.data(),      
		c_materials.data(),    
		static_cast<int>(numSpheres),
		cudaCamera,
		m_Settings.Accumulate,
		m_Settings.SkyBox
	);
	m_LastRayTraceTime = timer.ElapsedMillis();
#else
#if WL_RELEASE

	if (m_FrameIndex == 1)
		memset(m_AccumulationData, 0, m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(glm::vec4));
	
	Walnut::Timer timer;
	std::for_each(std::execution::par, m_ImageVerticalIter.begin(), m_ImageVerticalIter.end(),
		[this](uint32_t y)
		{
			std::for_each(std::execution::par, m_ImageHorizontalIter.begin(), m_ImageHorizontalIter.end(),
				[this, y](uint32_t x)
				{
					glm::vec4 color = PerPixel(x, y);
					m_AccumulationData[x + y * m_FinalImage->GetWidth()] += color;

					glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalImage->GetWidth()];
					accumulatedColor /= static_cast<float>(m_FrameIndex);

					accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));
					m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);  
				});
		});
	m_LastRayTraceTime = timer.ElapsedMillis();

#else

	ProsesUnit = "CPU";
	for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++)
	{
		for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++)
		{
			glm::vec4 color = PerPixel(x, y);
			m_AccumulationData[x + y * m_FinalImage->GetWidth()] += color;

			glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalImage->GetWidth()];
			accumulatedColor /= (float)m_FrameIndex;

			accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));
			m_ImageData[x + y * m_FinalImage->GetWidth()] = Utils::ConvertToRGBA(accumulatedColor);
		}
	}
#endif
#endif

	{
		Walnut::Timer timer;
		m_FinalImage->SetData(m_ImageData);
		m_LastSetDataTime = timer.ElapsedMillis();
	}
	
	if (m_Settings.Accumulate)
		m_FrameIndex++;
	else
		m_FrameIndex = 1;
}


inline glm::vec3 operator*(const glm::vec3& v, bool b) {
	return b ? v : glm::vec3(0.0f);
}

glm::vec4 Renderer::PerPixel(uint32_t x, uint32_t y)
{
	Ray ray;
	ray.Origin    = m_ActiveCamera->GetPosition();
	ray.Direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalImage->GetWidth()];

	glm::vec3 light(0.0f);
	glm::vec3 contrib(1.0f);

	int bounces = 50;
	for (int i = 0; i < bounces; i++)
	{
		auto payload = TraceRay(ray);
		if (payload.HitDistance < 0.0f)
		{
			// sky
			light += contrib * glm::vec3(0.6f, 0.7f, 0.9f) * m_Settings.SkyBox;
			break;
		}

		const Sphere&  sphere   = m_ActiveScene->Spheres[payload.ObjectIndex];
		const Material& material = m_ActiveScene->Materials[sphere.MaterialIndex];

		// accumulate emission
		light += contrib * material.GetEmission();

		// compute true normal
		glm::vec3 N = payload.WorldNormal;

		// diffuse hemisphere sampling (Lambertian)
		//glm::vec3 diffuseDir = glm::normalize(N + Walnut::Random::InUnitSphere()); 
		glm::vec3 diffuseDir = (N + Walnut::Random::InUnitSphere()); 
		// rough specular: perturb the microfacet normal by roughness
		//glm::vec3 perturbedN = glm::normalize(N + material.Roughness * Walnut::Random::InUnitSphere());
		glm::vec3 perturbedN = (N + material.Roughness * Walnut::Random::InUnitSphere());
		glm::vec3 specularDir = glm::reflect(ray.Direction, perturbedN);

		// blend between diffuse and specular by metallic
		//ray.Direction = glm::normalize(glm::mix(diffuseDir, specularDir, material.Metallic));
		ray.Direction = (glm::mix(diffuseDir, specularDir, material.Metallic));

		// energy conservation: metals have no diffuse; dielectrics reflect ~4%
		glm::vec3 F0 = glm::mix(glm::vec3(0.04f), material.Albedo, material.Metallic);
		glm::vec3 diffuseColor  = (1.0f - material.Metallic) * material.Albedo;
		glm::vec3 specularColor = F0;

		// update throughput
		contrib *= (diffuseColor + specularColor);

		ray.Origin = payload.WorldPosition + ray.Direction * 0.0001f;
	}

	return glm::vec4(light, 1.0f);
}

Renderer::HitPayload Renderer::TraceRay(const Ray& ray)
{
    int closestSphere = -1;
    float hitDistance = std::numeric_limits<float>::max();

    float a = glm::dot(ray.Direction, ray.Direction); // usually 1.0f
    float inv2a = 1.0f / (2.0f * a);

    for (size_t i = 0; i < m_ActiveScene->Spheres.size(); i++)
    {
        const Sphere& sphere = m_ActiveScene->Spheres[i];
        glm::vec3 origin = ray.Origin - sphere.Position;

        float b = 2.0f * glm::dot(origin, ray.Direction);
        float c = glm::dot(origin, origin) - sphere.Radius * sphere.Radius;

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f)
            continue;

        float sqrtDiscriminant = glm::sqrt(discriminant);
        float closestT = (-b - sqrtDiscriminant) * inv2a;
        if (closestT > 0.0f && closestT < hitDistance)
        {
            hitDistance = closestT;
            closestSphere = (int)i;
        }
    }

    if (closestSphere < 0)
        return Miss(ray);

    return ClosestHit(ray, hitDistance, closestSphere);
}


Renderer::HitPayload Renderer::ClosestHit(const Ray& ray, float hitDistance, int objectIndex)
{
	Renderer::HitPayload payload;
	payload.HitDistance = hitDistance;
	payload.ObjectIndex = objectIndex;

	const Sphere& closestSphere = m_ActiveScene->Spheres[objectIndex];

	glm::vec3 origin = ray.Origin - closestSphere.Position;
	payload.WorldPosition = origin + ray.Direction * hitDistance;
	payload.WorldNormal = glm::normalize(payload.WorldPosition);

	payload.WorldPosition += closestSphere.Position;

	return payload;
}

Renderer::HitPayload Renderer::Miss(const Ray& ray)
{
	Renderer::HitPayload payload;
	payload.HitDistance = -1.0f;
	return payload;
}
