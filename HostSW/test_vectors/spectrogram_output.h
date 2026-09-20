#ifndef SPECTROGRAM_OUTPUT_H
#define SPECTROGRAM_OUTPUT_H

#include <stdint.h>
#include "arm_math_types.h"  /* for float32_t */

#define SPECTROGRAM_OUTPUT_LEN (1350)

extern const float32_t spectrogram_output[SPECTROGRAM_OUTPUT_LEN];

#endif /* SPECTROGRAM_OUTPUT_H */
