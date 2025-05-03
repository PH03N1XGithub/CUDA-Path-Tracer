-- premake5.lua
workspace "PathTracing"
   architecture "x64"
   configurations { "Debug", "Release", "Dist","ReleaseGPU" }
   startproject "PathTracing"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "PathTracing"