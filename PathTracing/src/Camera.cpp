#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Walnut/Application.h"
#include "Walnut/Input/Input.h"

using namespace Walnut;

Camera::Camera()
{
	m_VerticalFOV = 20.0f;
	m_NearClip = 0.1f;
	m_FarClip = 1000.0f;
	//m_ForwardDirection = glm::vec3(-0.86, -0.33, -0.40);
	m_ForwardDirection = glm::vec3(0.97, -0.13, 0.24);
	//m_Position = glm::vec3(9.8, 4.6, 4.4);
	m_Position = glm::vec3(42.58, 3.02, 45.39);
	m_ViewportWidth = 1920;
	m_ViewportHeight = 1080;
	RecalculateView();
	RecalculateRayDirections();
	RecalculateProjection();
}

Camera::Camera(float verticalFOV, float nearClip, float farClip)
	: m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip)
{
	m_ForwardDirection = glm::vec3(-0.86, -0.33, -0.40);
	m_Position = glm::vec3(9.8, 4.6, 4.4);
	m_ViewportWidth = 1920;
	m_ViewportHeight = 1080;
	RecalculateView();
	RecalculateRayDirections();
	RecalculateProjection();
}

bool Camera::OnUpdate(float ts)
{
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(MouseButton::Right))
	{
		Input::SetCursorMode(CursorMode::Normal);
		return false;
	}
	
	Input::SetCursorMode(CursorMode::Locked);

	bool moved = false;

	constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);
	glm::vec3 rightDirection = glm::cross(m_ForwardDirection, upDirection);

	float speed = 5.0f;
	// Movement
	if (Input::IsKeyDown(KeyCode::LeftShift))
	{
		speed *= 4;
	}
	if (Input::IsKeyDown(KeyCode::W))
	{
		m_Position += m_ForwardDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::S))
	{
		m_Position -= m_ForwardDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::A))
	{
		m_Position -= rightDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::D))
	{
		m_Position += rightDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::Q))
	{
		m_Position -= upDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::E))
	{
		m_Position += upDirection * speed * ts;
		moved = true;
	}
	glm::vec2 scrollDelta = Walnut::Input::GetMouseScroll();
	if (scrollDelta.y < 0.0f)
	{
		if (m_FocusDistance > 1.0f)
		{
			m_FocusDistance--;
			moved = true;
		}
	}
	else if (scrollDelta.y > 0.0f)
	{
		m_FocusDistance++;
		moved = true;
	}
	
	if (Input::IsKeyDown(KeyCode::Z))
	{
		if (m_VerticalFOV >= 2.0f)
		{
			m_VerticalFOV -= 1.0f;
		}
		RecalculateProjection();
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::X))
	{
		m_VerticalFOV += 1.0f;
		RecalculateProjection();
		moved = true;
	}
	
	// Rotation
	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * m_RotationSpeed;
		float yawDelta = delta.x * m_RotationSpeed;

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));
		m_ForwardDirection = glm::rotate(q, m_ForwardDirection);

		moved = true;
	}

	if (moved)
	{
		RecalculateView();
		RecalculateRayDirections();
	}

	return moved;
}

void Camera::OnResize(const uint32_t width, const uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
		return;

	m_ViewportWidth = width;
	m_ViewportHeight = height;

	RecalculateProjection();
	RecalculateRayDirections();
}



void Camera::RecalculateProjection()
{
	m_Projection = glm::perspectiveFov(glm::radians(m_VerticalFOV), static_cast<float>(m_ViewportWidth), static_cast<float>(m_ViewportHeight), m_NearClip, m_FarClip);
	m_InverseProjection = glm::inverse(m_Projection);
}

void Camera::RecalculateView()
{
	m_View = glm::lookAt(m_Position, m_Position + m_ForwardDirection, glm::vec3(0, 1, 0));
	m_InverseView = glm::inverse(m_View);
}

void Camera::RecalculateRayDirections()
{
	m_RayDirections.resize(m_ViewportWidth * m_ViewportHeight);

	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			glm::vec2 coord = { static_cast<float>(x) / static_cast<float>(m_ViewportWidth), static_cast<float>(y) / static_cast<float>(m_ViewportHeight) };
			coord = coord * 2.0f - 1.0f; // -1 -> 1

			glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
			glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
			m_RayDirections[x + y * m_ViewportWidth] = rayDirection;
		}
	}
}

CudaCamera Camera::GetGetCudaCamera(const Camera& camera, uint32_t width, uint32_t height) {
	CudaCamera cudaCamera;
	glm::mat4 view = camera.GetView();

	// Position and direction of the camera
	cudaCamera.Position = make_float3(camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z);
	cudaCamera.ForwardDirection = make_float3(camera.GetDirection().x, camera.GetDirection().y, camera.GetDirection().z);

	// Up and Right vectors
	cudaCamera.Up = make_float3(view[1][0], view[1][1], view[1][2]);
	cudaCamera.Right = make_float3(view[0][0], view[0][1], view[0][2]);

	// Camera settings
	cudaCamera.FovY = camera.m_VerticalFOV * (PI / DEGREES180);
	cudaCamera.AspectRatio = static_cast<float>(width) / static_cast<float>(height);
	cudaCamera.NearClip = camera.m_NearClip;
	cudaCamera.FarClip = camera.m_FarClip;
	cudaCamera.Aperture = camera.m_Aperture;
	cudaCamera.FocusDistance = camera.m_FocusDistance;

	// Copy matrices
	cudaCamera.ViewMatrix = ConvertMat4ToFloat4x4(camera.GetView());
	cudaCamera.InverseViewMatrix = ConvertMat4ToFloat4x4(camera.GetInverseView());
	cudaCamera.ProjectionMatrix = ConvertMat4ToFloat4x4(camera.GetProjection());
	cudaCamera.InverseProjectionMatrix = ConvertMat4ToFloat4x4(camera.GetInverseProjection());

	return cudaCamera;
}

float4x4 Camera::ConvertMat4ToFloat4x4(const glm::mat4& mat)
{
	float4x4 result;
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
		{
			result.m[row][col] = mat[col][row];
		}
	}
	return result;
}
