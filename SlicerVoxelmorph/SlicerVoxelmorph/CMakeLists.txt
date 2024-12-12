cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerVoxelmorph)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerVoxelmorph")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerVoxelmorph.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SlicerVoxelmorph/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Voxelmorph)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
