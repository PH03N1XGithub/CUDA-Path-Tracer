#include "Input.h"

#include "Walnut/Application.h"

#include <GLFW/glfw3.h>

namespace Walnut {

	bool Input::IsKeyDown(KeyCode keycode)
	{
		GLFWwindow* windowHandle = Application::Get().GetWindowHandle();
		int state = glfwGetKey(windowHandle, (int)keycode);
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool Input::IsMouseButtonDown(MouseButton button)
	{
		GLFWwindow* windowHandle = Application::Get().GetWindowHandle();
		int state = glfwGetMouseButton(windowHandle, (int)button);
		return state == GLFW_PRESS;
	}

	glm::vec2 Input::GetMousePosition()
	{
		GLFWwindow* windowHandle = Application::Get().GetWindowHandle();

		double x, y;
		glfwGetCursorPos(windowHandle, &x, &y);
		return { (float)x, (float)y };
	}
	glm::vec2 Input::GetMouseScroll()
	{
		GLFWwindow* windowHandle = Application::Get().GetWindowHandle();
		glfwSetScrollCallback(windowHandle, Walnut::Input::GLFWScrollCallback);
		glm::vec2 delta = s_MouseScroll;
		s_MouseScroll = glm::vec2(0.0f); // Reset after read
		return delta;
	}

	void Input::GLFWScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
	{
		window = Application::Get().GetWindowHandle();
		s_MouseScroll = glm::vec2((float)xOffset, (float)yOffset);
	}

	void Input::SetCursorMode(CursorMode mode)
	{
		GLFWwindow* windowHandle = Application::Get().GetWindowHandle();
		glfwSetInputMode(windowHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL + (int)mode);
	}

}