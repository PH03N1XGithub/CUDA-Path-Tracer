#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "Walnut/Random.h"

struct Material
{
	glm::vec3 Albedo{ 1.0f };
	float Roughness = 1.0f;
	float Metallic = 0.0f;
	glm::vec3 EmissionColor{ 0.0f };
	float EmissionPower = 0.0f;

	glm::vec3 GetEmission() const { return EmissionColor * EmissionPower; }


	explicit operator CudaMaterial() const {
		CudaMaterial result;
		result.Albedo = make_float3(Albedo.x, Albedo.y, Albedo.z);
		result.Emission = make_float3(GetEmission().x, GetEmission().y, GetEmission().z);
		result.Roughness = Roughness;
		result.Metallic = Metallic;
		return result;
	}
};

struct Sphere
{
	glm::vec3 Position{0.0f};
	float Radius = 0.5f;

	int MaterialIndex = 0;

	explicit operator CudaSphere() const {
		CudaSphere result;
		result.Position = make_float3(Position.x, Position.y, Position.z);
		result.Radius = Radius;
		result.MaterialIndex = MaterialIndex;
		return result;
	}
};

struct Scene
{
	Scene();
	std::vector<Sphere> Spheres;
	std::vector<Material> Materials;
};

inline Scene::Scene()
{
	Material& pinkSphere = Materials.emplace_back();
	pinkSphere.Albedo = { 1.0f, 0.56f, 0.0f };
	pinkSphere.Roughness = 1.0f;
	pinkSphere.Metallic = 0.0f;

	Material& blueSphere = Materials.emplace_back();
	blueSphere.Albedo = { 1.0f, 0.58f, 0.0f };
	blueSphere.Roughness = 1.0f;
	blueSphere.Metallic = 0.0f;

	Material& orangeSphere = Materials.emplace_back();
	orangeSphere.Albedo = { 0.5f, 0.4f, 0.0f };
	orangeSphere.Roughness = 0.1f;
	orangeSphere.EmissionColor = orangeSphere.Albedo;
	orangeSphere.EmissionPower = 10.0f;

	Material& RedSphere = Materials.emplace_back();
	RedSphere.Albedo = { 0.8f, 0.8f, 0.8f };
	RedSphere.Roughness = 0.0f;
	RedSphere.Metallic = 1.0f;
	RedSphere.EmissionColor = RedSphere.Albedo;
	RedSphere.EmissionPower = 0.0f;

	

	{
		Sphere sphere;
		sphere.Position = { 0.0f, 1.0f, 0.0f };
		sphere.Radius = 2.0f;
		sphere.MaterialIndex = 0;
		Spheres.push_back(sphere);
	}
	{
		Sphere sphere;
		sphere.Position = { 30.0f, -0.9f, -30.0f };
		sphere.Radius = 15.0f;
		sphere.MaterialIndex = 2;
		Spheres.push_back(sphere);
	}
	{
		Sphere sphere;
		sphere.Position = { 50.1f, 2.0f, 52.0f };
		sphere.Radius = 2.0f;
		sphere.MaterialIndex = 3;
		Spheres.push_back(sphere);
	}
	{
		Sphere sphere;
		sphere.Position = { 0.0f, -101.0f, 0.0f };
		sphere.Radius = 0.0f;
		sphere.MaterialIndex = 1;
		Spheres.push_back(sphere);
	}


	for (int i = 0; i < 20 * 20; i++)
	{
		int x = i % 20;
		int z = i / 20;

		Sphere sphere;
		sphere.Position = { (x + Walnut::Random::Float()) * 4.5f + 20 , 0.5f, (z + Walnut::Random::Float())* 4.5f +20 };
		sphere.Radius = Walnut::Random::Float() * 1.5f;
		sphere.Position.y = sphere.Radius;

		// Create a unique material
		Material material;
		material.Albedo = glm::vec3(Walnut::Random::Float(), Walnut::Random::Float(), Walnut::Random::Float());
		material.Roughness = Walnut::Random::Float();      // [0, 1]
		material.Metallic = Walnut::Random::Float() < 0.5f ? 0.0f : 1.0f; // 50% chance of being metallic

		// Store material and assign index
		int materialIndex = static_cast<int>(Materials.size());
		Materials.push_back(material);
		sphere.MaterialIndex = materialIndex;

		Spheres.push_back(sphere);
	}
}