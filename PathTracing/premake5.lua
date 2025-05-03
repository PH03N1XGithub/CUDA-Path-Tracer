project "RayTracing"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "src/**.h", "src/**.cpp" }

   -- Manually handle CUDA files
   files { "src/**.cu" }
  

   includedirs
   {
      "../Walnut/vendor/imgui",
      "../Walnut/vendor/glfw/include",
      "../Walnut/vendor/glm",

      "../Walnut/Walnut/src",

      "%{IncludeDir.VulkanSDK}",
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
   }

   libdirs {
      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
   }

   links
   {
       "Walnut",
       "cudart"
   }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "files:**.cu"
    buildmessage "Compiling CUDA file: %{file.relpath}"
    buildcommands {
        '"$(CUDA_PATH)/bin/nvcc" -c -Xcompiler "/EHsc /W3 /nologo /O2 /MD" ' ..
        '-gencode=arch=compute_86,code=sm_86 ' .. -- Change 86 to match your GPU
        '"%{file.relpath}" -o "%{cfg.objdir}/%{file.basename}.obj"'
    }
    buildoutputs {
        "%{cfg.objdir}/%{file.basename}.obj"
    }


   files { "../bin-int/" .. outputdir .. "/%{prj.name}/cuda.obj" }



   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG" }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE" }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:ReleaseGPU"
      defines { "WL_RELEASE_GPU" }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST" }
      runtime "Release"
      optimize "On"
      symbols "Off"