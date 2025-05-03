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
		ImGui::Begin("Settings");
		ImGui::Text("FPS: %.3f", 1000/m_LastRenderTime);
		ImGui::Text("Last render: %.3fms", m_LastRenderTime);
		ImGui::Text((m_Renderer.m_ProsesUnit + " Time: %.3fms").c_str(), m_Renderer.m_LastRayTraceTime);
		ImGui::Text("Last SetFrameData: %.3fms", m_Renderer.m_LastSetDataTime);
		ImGui::Text("Accumulation Data: %d", m_Renderer.m_FrameIndex);
		glm::vec3 cameraDir = m_Camera.GetDirection();
		glm::vec3 cameraPos = m_Camera.GetPosition();
		ImGui::Text("Camera direction: (%.2f, %.2f, %.2f)",cameraDir.x, cameraDir.y, cameraDir.z);
		ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)",cameraPos.x, cameraPos.y, cameraPos.z);

		
		if (ImGui::Button("Add Sphere"))
		{
			constexpr Sphere sphere;
			m_Scene.Spheres.push_back(sphere);
		}
		if (ImGui::Button("Add Material"))
		{
			Material& red_sphere = m_Scene.Materials.emplace_back();
		}
		
		ImGui::Checkbox("Accumulate", &m_Renderer.GetSettings().Accumulate);
		ImGui::Checkbox("SkyBox", &m_Renderer.GetSettings().SkyBox);
		
		if (ImGui::Button("Reset"))
			m_Renderer.ResetFrameIndex();

		ImGui::End();

		ImGui::Begin("Scene");
		for (size_t i = 0; i < m_Scene.Spheres.size(); i++)
		{
			ImGui::PushID(i);

			Sphere& sphere = m_Scene.Spheres[i];
			ImGui::Text("Sphere %d", i);
			if (ImGui::DragFloat3("Position", glm::value_ptr(sphere.Position), 0.1f) ||
				ImGui::DragFloat("Radius", &sphere.Radius, 0.1f) ||
				ImGui::DragInt("Material", &sphere.MaterialIndex, 1.0f, 0, (int)m_Scene.Materials.size() - 1))
			{
				m_Renderer.ResetFrameIndex();
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
				m_Renderer.ResetFrameIndex();
			}
			

			ImGui::Separator();

			ImGui::PopID();
		}
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");

		m_ViewportWidth = ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = ImGui::GetContentRegionAvail().y;

		if (const auto image = m_Renderer.GetFinalImage())
			ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() },
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

	float m_LastRenderTime = 0.0f;
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