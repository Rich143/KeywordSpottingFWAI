#include <float.h>
#include "unity.h"

#include "audio_preprocessing.h"
#include "spectrogram_signal_input.h"
#include "spectrogram_output.h"

#include "test_helpers.h"

#include "mel_filterbank.h"

#include "arm_math.h"

#define DEBUG_DUMP_FILTERED_OUTPUT 1
#if DEBUG_DUMP_FILTERED_OUTPUT
#include <stdio.h>
#endif

#define TEST_PREPROC_OK(expr) \
    do { \
        audio_preprocessing_status_t status = (expr); \
        TEST_ASSERT_EQUAL_INT_MESSAGE(AUDIO_PREPROCESSING_STATUS_OK, status, "Expected AUDIO_PREPROCESSING_STATUS_OK"); \
    } while (0)

#define TEST_PREPROC_STATUS(expr, expected) \
    do { \
        audio_preprocessing_status_t status = (expr); \
        TEST_ASSERT_EQUAL_INT_MESSAGE((expected), status, "Unexpected audio_preprocessing_status_t value"); \
    } while (0)

// To get these directives to work:
// - quote the file name in " "
// - Add the directory containing the file to :paths: in project.yml
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_rfft_fast_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c")

// These files are referenced by rfft files above
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_init_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_bitreversal2.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/CommonTables/arm_const_structs.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/CommonTables/arm_common_tables.c")

// Audio preproc source files
TEST_SOURCE_FILE("../Middlewares/ST/STM32_AI_AudioPreprocessing_Library/Src/common_tables.c")
TEST_SOURCE_FILE("../Middlewares/ST/STM32_AI_AudioPreprocessing_Library/Src/feature_extraction.c")
TEST_SOURCE_FILE("../Middlewares/ST/STM32_AI_AudioPreprocessing_Library/Src/mel_filterbank.c")
TEST_SOURCE_FILE("../Middlewares/ST/STM32_AI_AudioPreprocessing_Library/Src/window.c")
TEST_SOURCE_FILE("../Middlewares/ST/STM32_AI_AudioPreprocessing_Library/Src/dct.c")

// Other CMSIS-DSP source files needed by Audio preproc Lib
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/ComplexMathFunctions/arm_cmplx_mag_squared_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/BasicMathFunctions/arm_mult_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/BasicMathFunctions/arm_scale_f32.c")
TEST_SOURCE_FILE("../Drivers/CMSIS-DSP/Source/BasicMathFunctions/arm_offset_f32.c")

void setUp(void) {
    TEST_PREPROC_OK(audio_preprocessing_init());
}

void tearDown(void) {}

void process_frame_and_check_result(float32_t *pInSignal) {
    float32_t pInSignalCopy[AUDIO_SPECTROGRAM_FRAME_LEN];

    memcpy(pInSignalCopy, pInSignal, AUDIO_SPECTROGRAM_FRAME_LEN * sizeof(float32_t));
    TEST_PREPROC_OK(audio_preprocessing_process_frame(pInSignalCopy));
}

void test_preproc(void) {
    for (int i = 0; i < AUDIO_SPECTROGRAM_COLS; i++) {
        process_frame_and_check_result(spectrogram_signal_input);
    }

    float32_t *spectrogram = audio_preprocessing_get_spectrogram();
    uint32_t spectrogram_len = audio_preprocessing_get_spectrogram_len();

#if DEBUG_DUMP_FILTERED_OUTPUT
    // Print the spectrogram
    for (uint32_t i = 0; i < spectrogram_len; i++) {
        printf("%.7g ", spectrogram[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
    printf("\n");
#endif

    check_signal_close(spectrogram, spectrogram_output, spectrogram_len, 1e-4f);
}

void test_single_col(void) {

    process_frame_and_check_result(spectrogram_signal_input);
    
    TEST_ASSERT_EQUAL_MESSAGE(1, audio_preprocessing_get_spectrogram_filled_cols(), "Only one column should be filled");
}

void test_two_cols(void) {
    float32_t pInSignalCopy[AUDIO_SPECTROGRAM_FRAME_LEN];

    process_frame_and_check_result(spectrogram_signal_input);

    process_frame_and_check_result(spectrogram_signal_input);
    
    TEST_ASSERT_EQUAL_MESSAGE(2, audio_preprocessing_get_spectrogram_filled_cols(), "Two columns should be filled");
}

void test_full_spectrogram(void) {
    for (int i = 0; i < AUDIO_SPECTROGRAM_COLS; i++) {
        process_frame_and_check_result(spectrogram_signal_input);
    }
    
    TEST_ASSERT_EQUAL_MESSAGE(AUDIO_SPECTROGRAM_COLS, audio_preprocessing_get_spectrogram_filled_cols(), "All columns should be filled");
}

void test_add_to_full_spectrogram(void) {

    for (int i = 0; i < AUDIO_SPECTROGRAM_COLS; i++) {
        process_frame_and_check_result(spectrogram_signal_input);
    }
    TEST_ASSERT_EQUAL_MESSAGE(AUDIO_SPECTROGRAM_COLS, audio_preprocessing_get_spectrogram_filled_cols(), "All columns should be filled");

    float32_t pInSignalCopy[AUDIO_SPECTROGRAM_FRAME_LEN];
    memcpy(pInSignalCopy, spectrogram_signal_input, AUDIO_SPECTROGRAM_FRAME_LEN * sizeof(float32_t));

    TEST_PREPROC_STATUS(AUDIO_PREPROCESSING_STATUS_ERROR_SPECTROGRAM_FULL, audio_preprocessing_process_frame(pInSignalCopy));

    TEST_ASSERT_EQUAL_MESSAGE(AUDIO_SPECTROGRAM_COLS, audio_preprocessing_get_spectrogram_filled_cols(), "All columns should be filled");
}

void test_clear_full_spectrogram(void) {
    for (int i = 0; i < AUDIO_SPECTROGRAM_COLS; i++) {
        process_frame_and_check_result(spectrogram_signal_input);
    }
    
    TEST_ASSERT_EQUAL_MESSAGE(AUDIO_SPECTROGRAM_COLS, audio_preprocessing_get_spectrogram_filled_cols(), "All columns should be filled");

    TEST_PREPROC_OK(audio_preprocessing_clear_spectrogram());

    TEST_ASSERT_EQUAL_MESSAGE(0, audio_preprocessing_get_spectrogram_filled_cols(), "No columns should be filled");
}
