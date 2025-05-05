#pragma once

#include "KeyCodes.h"

#include <glm/glm.hpp>

#include "GLFW/glfw3.h"

namespace Walnut {

	class Input
	{
	public:
		static bool IsKeyDown(KeyCode keycode);
		static bool IsMouseButtonDown(MouseButton button);

		static glm::vec2 GetMousePosition();

		static void SetCursorMode(CursorMode mode);

		static glm::vec2 GetMouseScroll(); // NEW

		// Internal use only
		static void GLFWScrollCallback(GLFWwindow* window, double xOffset, double yOffset);

	private:
		static inline glm::vec2 s_MouseScroll = glm::vec2(0.0f);
	};

}
