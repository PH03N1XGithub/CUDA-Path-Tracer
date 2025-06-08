# CUDA GPU-Accelerated Real-Time Path Tracer

A GPU-accelerated real-time path tracer built with CUDA and C++ for realistic Path tracing, simulating physically based rendering (PBR) materials, accurate lighting, shadows, and reflections.

## Building and running
Officially supports Windows 10/11 and Visual Studio 2022 You'll need to have the [Vulkan SDK](https://vulkan.lunarg.com/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)installed.

1. Clone recursively: `git clone --recursive https://github.com/PH03N1XGithub/CUDA-Path-Tracer.git`
2. Run `scripts/Setup.bat`
3. Open `PathTracing.sln` and hit F5 (Change configuration to ReleaseGPU)

## Features
Real-Time Path Tracing: Leverages GPU parallelism using CUDA for efficient and fast ray tracing.

Physically Based Rendering (PBR): Implements PBR material models for accurate simulation of surface interactions, including roughness and metalness.

Global Illumination: Simulates realistic indirect lighting with multiple bounces.

Soft Shadows & Reflections: Realistic shadows and reflections based on physical light transport.

Camera Focus / Autofocus (AF): Dynamically simulates camera lens focusing with depth of field effects to enhance realism.

Multiple Material Support: Supports diffuse, reflective, and refractive materials.

Configurable Samples per Pixel: Adjustable sample count for balancing quality and performance.

![PathTracer](https://github.com/user-attachments/assets/9ac7d68b-6ed7-4a8a-aefd-da76999d7b0e)

![PathTracerDark](https://github.com/user-attachments/assets/e6a5fe13-9282-4e7b-934c-f7c32d66389c)

![Ekran görüntüsü 2025-06-08 210313](https://github.com/user-attachments/assets/49a1c3aa-196a-47df-bf28-1918268c6fad)

![Ekran görüntüsü 2025-06-08 211814](https://github.com/user-attachments/assets/7bac634b-78ee-4f37-b11b-c1258a95b981)

![Ekran görüntüsü 2025-06-08 213227](https://github.com/user-attachments/assets/7a17145b-980d-48ad-8dca-cafb1c9d50f6)
