cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME B-U585I-IOT02A)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    Core/Src/audio.c
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
    CMSISDSP::CMSISDSP
    STM32AudioProc
    AudioPreprocessing

    # Drivers
    b-u585i-iot02a-bsp
    stm32-bsp-common
)
# CMakeLists.txt for STM32Cube AI Studio
include(AI/STMicroelectronics.Middlewares.cmake)
# CMakeLists.txt for STM32Cube AI Studio
include(AI/STMicroelectronics.STM32CubeAIStudio.cmake)

# CMSIS-DSP
set(CMSISCORE "${CMAKE_SOURCE_DIR}/Drivers/CMSIS" CACHE PATH "")
set(CMSISDSP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(Middlewares/ARM/CMSIS-DSP/Source cmsis-dsp-build)

# Middlewares
add_subdirectory(Middlewares/ST/STM32_AI_AudioPreprocessing_Library)

# Lib
add_subdirectory(Lib)

# Drivers
add_subdirectory(Drivers)
