cmake_minimum_required(VERSION 3.14)
project(kws_host_lib C)

set(HOST ON CACHE BOOL "" FORCE)
set(CMSISDSP_INSTALL OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Middlewares/ARM/CMSIS-DSP/Source cmsis-dsp-build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Middlewares/ST/STM32_AI_AudioPreprocessing_Library stm32audioproc-build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Lib/AudioPreprocessing audioproc-build)

target_compile_definitions(STM32AudioProc PRIVATE __INLINE=inline)
