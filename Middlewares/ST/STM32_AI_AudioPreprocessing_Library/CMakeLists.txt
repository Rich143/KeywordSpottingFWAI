add_library(STM32AudioProc STATIC
    Src/common_tables.c
    Src/dct.c
    Src/feature_extraction.c
    Src/mel_filterbank.c
    Src/window.c
    Inc/common_tables.h
    Inc/dct.h
    Inc/feature_extraction.h
    Inc/mel_filterbank.h
    Inc/window.h
)

target_link_libraries(STM32AudioProc PUBLIC
    CMSISDSP::CMSISDSP)


target_include_directories(STM32AudioProc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc
)
