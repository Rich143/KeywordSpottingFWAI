add_library(AudioPreprocessing
    audio_preprocessing.c
    audio_preprocessing.h
)

target_link_libraries(AudioPreprocessing PRIVATE
    STM32AudioProc)

target_include_directories(AudioPreprocessing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
