#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Timer.h"

#include "Renderer.h"
#include "Camera.h"

#include <glm/gtc/type_ptr.hpp>

using namespace Walnut;


class ExampleLayer : public Walnut::Layer
{
public:


	virtual void OnUpdate(float ts) override
	{
		if (m_Camera.OnUpdate(ts))
			m_Renderer.ResetFrameIndex();
	}

	virtual void OnUIRender() override
	{
		ImGui::StyleColorsDark();
		ImGui::GetStyle().Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // White text
		ImGui::GetStyle().Colors[ImGuiCol_Border] = ImVec4(0.3f, 0.3f, 0.0f, 1.0f);  // Dark borders
		ImGui::GetStyle().Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.6f, 0.0f, 1.0f);    // Custom button color
		ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.7f, 0.0f, 1.0f); // Hovered button color
		ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = ImVec4(0.4f, 0.8f, 0.0f, 1.0f); // Active button color
		ImGui::GetStyle().Colors[ImGuiCol_CheckMark] = ImVec4(0.4f, 0.8f, 0.0f, 1.0f); // Active button color
		ImGui::GetStyle().Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.4f, 0.8f, 0.0f, 1.0f); // Active button color
		ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive] = ImVec4(0.4f, 0.8f, 0.0f, 1.0f); // Active button color
		ImGui::GetStyle().Colors[ImGuiCol_Separator] = ImVec4(0.2f, 0.2f, 0.0f, 1.0f);  // Separator color
		ImGui::GetStyle().Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.4f, 0.4f, 0.0f, 1.0f); // Separator hover color
		ImGui::GetStyle().Colors[ImGuiCol_SeparatorActive] = ImVec4(0.6f, 0.6f, 0.0f, 1.0f); // Separator active color
		ImGui::GetStyle().Colors[ImGuiCol_Header] = ImVec4(0.2f, 0.5f, 0.0f, 1.0f); // Header background color
		ImGui::GetStyle().Colors[ImGuiCol_HeaderHovered] = ImVec4(0.3f, 0.6f, 0.0f, 1.0f); // Hovered header color
		ImGui::GetStyle().Colors[ImGuiCol_HeaderActive] = ImVec4(0.4f, 0.7f, 0.0f, 1.0f);  // Active header color
		ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);  // Input box background
		ImGui::GetStyle().Colors[ImGuiCol_Tab] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_TabActive] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_TabHovered] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_TabUnfocused] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_MenuBarBg] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_TitleBgActive] = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);  
		ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 0.3f;  // Full transparency
		ImGui::GetStyle().WindowRounding = 5.0f; // Rounded window corners
		ImGui::GetStyle().FrameRounding = 3.0f;  // Rounded frame corners for buttons, inputs, etc.
		

		if (ImGui::BeginMenuBar())
		{
			ImGui::SetNextWindowBgAlpha(0.3f);
			bool bShouldReset = false;
			// Begin a menu within the menu bar
			if (ImGui::BeginMenu("Settings")) 
			{
				ImGui::Text("FPS: %.3f", 1000/m_LastRenderTime);
				ImGui::Text("Last render: %.3fms", m_LastRenderTime);
				ImGui::Text((m_Renderer.ProsesUnit + " Time: %.3fms").c_str(), m_Renderer.LastRayTraceTime);
				ImGui::Text("CPU: %.3fms", m_Renderer.LastSetDataTime);
				//ImGui::Text("Accumulation Data: %d", m_Renderer.m_FrameIndex);
				const glm::vec3 cameraDir = m_Camera.GetDirection();
				const glm::vec3 cameraPos = m_Camera.GetPosition();
				ImGui::Text("Camera direction: (%.2f, %.2f, %.2f)",cameraDir.x, cameraDir.y, cameraDir.z);
				ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)",cameraPos.x, cameraPos.y, cameraPos.z);
				ImGui::Text("Object Count: %d", m_Scene.Spheres.size());
			
				if (ImGui::Button("Add Sphere"))
					m_Scene.Spheres.emplace_back();
				
				if (ImGui::Button("Add Material"))
					m_Scene.Materials.emplace_back();
				
			
				ImGui::InputInt("Max Bounce", &m_Renderer.GetSettings().maxBounces);
			
				if (ImGui::InputInt("SPP", &m_Renderer.GetSettings().samplesPerPixel) && m_Renderer.GetSettings().samplesPerPixel == 0)
					m_Renderer.GetSettings().samplesPerPixel = 1;
				
				if (ImGui::DragFloat("Aperture",&m_Camera.GetAperture(),0.01f,0,0.3f))
					bShouldReset = true;
				
				if (ImGui::DragFloat("FocusDistance", &m_Camera.GetFocusDistance(),0.1f,1.0f,FLT_MAX))
					bShouldReset = true;
				
			
				ImGui::Checkbox("Accumulate", &m_Renderer.GetSettings().Accumulate);
				if (ImGui::Checkbox("SkyBox", &m_Renderer.GetSettings().SkyBox))
					bShouldReset = true;
			
				if (ImGui::Button("Reset"))
					m_Renderer.ResetFrameIndex();
				

				ImGui::EndMenu();
			}
			ImGui::SetNextWindowBgAlpha(0.3f);
			if (ImGui::BeginMenu("Scene Inspector"))
			{

				const ImVec2 size = ImVec2(430, 450);
				ImGui::BeginChild("Scene Inspector", size, false);
				
				for (size_t i = 0; i < m_Scene.Spheres.size(); i++)
				{
					ImGui::PushID(i);
					Sphere& sphere = m_Scene.Spheres[i];
					ImGui::Text("Sphere %d", i);
					if (ImGui::DragFloat3("Position", glm::value_ptr(sphere.Position), 0.1f) ||
						ImGui::DragFloat("Radius", &sphere.Radius, 0.1f) ||
						ImGui::DragInt("Material", &sphere.MaterialIndex, 1.0f, 0, (int)m_Scene.Materials.size() - 1))
					{
						bShouldReset = true;
					}
					ImGui::Separator();
					ImGui::PopID();
				}
		
				ImGui::Separator();
				ImGui::Separator();
		
				for (size_t i = 0; i < m_Scene.Materials.size(); i++)
				{
					ImGui::PushID(i);
					Material& material = m_Scene.Materials[i];
					ImGui::Text("Material %d", i);
					if (ImGui::ColorEdit3("Albedo", glm::value_ptr(material.Albedo))||
						ImGui::DragFloat("Roughness", &material.Roughness, 0.05f, 0.0f, 1.0f)||
						ImGui::DragFloat("Metallic", &material.Metallic, 0.05f, 0.0f, 1.0f)||
						ImGui::ColorEdit3("Emission Color", glm::value_ptr(material.EmissionColor))||
						ImGui::DragFloat("Emission Power", &material.EmissionPower, 0.05f, 0.0f, FLT_MAX))
					{
						bShouldReset = true;
					}
					ImGui::Separator();
					ImGui::PopID();
				}
				ImGui::EndChild();
				ImGui::EndMenu();
			}
			ImGui::Text("FPS: %.f", 1000/m_LastRenderTime); // Display FPS in the menu bar
			// End the menu bar
			ImGui::EndMenuBar();
			if (bShouldReset)
				m_Renderer.ResetFrameIndex();
		}

		
		
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		const ImVec2 windowSize = ImVec2(800, 600); // Width, Height
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);
		ImGui::Begin("Viewport");

		m_ViewportWidth = ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = ImGui::GetContentRegionAvail().y;

		if (const auto image = m_Renderer.GetFinalImage())
			ImGui::Image(image->GetDescriptorSet(), { static_cast<float>(image->GetWidth()), static_cast<float>(image->GetHeight()) },
				ImVec2(0, 1), ImVec2(1, 0));

		ImGui::End();
		ImGui::PopStyleVar();

		Render();
	}

	void Render()
	{
		Timer timer;
		
		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_Renderer.Render(m_Scene, m_Camera);

		m_LastRenderTime = timer.ElapsedMillis();
	}
private:
	Renderer m_Renderer;
	Camera m_Camera;
	Scene m_Scene;
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	float m_LastRenderTime = 0;
};



Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Ray Tracing";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}