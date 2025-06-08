#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <vector_functions.h>
#include "CudaStructs.h"

#define PI  3.14159265358979323846f
#define DEGREES180  180.f

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

private:
	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirections();
	

private:
	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	float m_VerticalFOV = 20.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	glm::vec3 m_Position{0.0f, 0.0f, 0.0f};
	glm::vec3 m_ForwardDirection{0.0f, 0.0f, 0.0f};

	// Cached ray directions
	std::vector<glm::vec3> m_RayDirections;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	size_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	float m_Aperture = 0.1f;         // Controls blur intensity
	float m_FocusDistance = 10.0f;   // Distance to the focus plane
	float m_RotationSpeed = 0.6f;

public:

	float& GetAperture()  { return m_Aperture; }
	float& GetFocusDistance()  { return m_FocusDistance; }

	static CudaCamera GetGetCudaCamera(const Camera& camera, uint32_t width, uint32_t height);
	static float4x4 ConvertMat4ToFloat4x4(const glm::mat4& mat);
	
};
