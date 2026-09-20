#include <float.h>
#include "unity.h"

#include "test_signal_input.h"
#include "test_signal_fft_output.h"
#include "test_helpers.h"

#include "arm_math.h"

#define DEBUG_DUMP_FILTERED_OUTPUT 0
#if DEBUG_DUMP_FILTERED_OUTPUT
#include <stdio.h>
#endif

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

arm_rfft_fast_instance_f32 S; // RFFT instance.

void setUp(void) {
    arm_status status;
    status = arm_rfft_fast_init_512_f32(&S);
}
void tearDown(void) {}

/*
p size      : nfft   reals
pOut size   : nfft   reals (nfft/2 complex)
*/
const uint32_t output_len = 512;

static float32_t testOutput[output_len];

void test_fft(void) {
    arm_rfft_fast_f32(&S, test_signal_input, testOutput, 0);

    check_signal_close(testOutput, test_signal_fft_output, output_len, 1e-4f);

#if DEBUG_DUMP_FILTERED_OUTPUT
    /* Print the output signal */
    for (uint32_t i = 0; i < output_len; i += 6) {
        printf("%.7g %.7g %.7g %.7g %.7g %.7g\n",
               testOutput[i], testOutput[i + 1], testOutput[i + 2],
               testOutput[i + 3], testOutput[i + 4], testOutput[i + 5]);
    }
#endif
}
