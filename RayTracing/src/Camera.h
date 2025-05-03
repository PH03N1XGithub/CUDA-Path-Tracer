#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <vector_functions.h>
#include "CudaStructs.h"

class Camera
{
public:
	Camera();
	Camera(float verticalFOV, float nearClip, float farClip);

	bool OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	const glm::mat4& GetProjection() const { return m_Projection; }
	const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	const glm::mat4& GetView() const { return m_View; }
	const glm::mat4& GetInverseView() const { return m_InverseView; }
	
	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetDirection() const { return m_ForwardDirection; }

	const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirections; }

	float GetRotationSpeed();
private:
	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirections();
private:
	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	float m_VerticalFOV = 45.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	glm::vec3 m_Position{0.0f, 0.0f, 0.0f};
	glm::vec3 m_ForwardDirection{0.0f, 0.0f, 0.0f};

	// Cached ray directions
	std::vector<glm::vec3> m_RayDirections;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

public:
	static CudaCamera GetGetCudaCamera(const Camera& camera, uint32_t width, uint32_t height) {
		CudaCamera cudaCamera;
		glm::mat4 view = camera.GetView();

		// Position and direction of the camera
		cudaCamera.Position = make_float3(camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z);
		cudaCamera.ForwardDirection = make_float3(camera.GetDirection().x, camera.GetDirection().y, camera.GetDirection().z);

		// Up and Right vectors
		cudaCamera.Up = make_float3(view[1][0], view[1][1], view[1][2]);
		cudaCamera.Right = make_float3(view[0][0], view[0][1], view[0][2]);

		// Camera settings
		cudaCamera.FovY = camera.m_VerticalFOV;
		cudaCamera.AspectRatio = static_cast<float>(width) / static_cast<float>(height);
		cudaCamera.NearClip = camera.m_NearClip;
		cudaCamera.FarClip = camera.m_FarClip;

		return cudaCamera;
	}

};
