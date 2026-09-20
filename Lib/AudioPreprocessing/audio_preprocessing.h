#pragma once

#include "arm_math_types.h"

// At 16000 Hz, 512 = (1/(16000))*512*1000 = 32 ms
// Choose this to be a power of 2 so that the FFT is also a power of 2
#define AUDIO_SPECTROGRAM_FRAME_LEN 512 // (32 ms)
#define AUDIO_SPECTROGRAM_STRIDE_LEN 352 // (22 ms)
#define AUDIO_SAMPLE_RATE_HZ        16000

#define AUDIO_SPECTROGRAM_NMELS     30
#define AUDIO_SPECTROGRAM_ROWS      AUDIO_SPECTROGRAM_NMELS
// Frame len = 512 samples (32 ms), stride len is 352 (22 ms), giving 45
// columns per 1 second clip
#define AUDIO_SPECTROGRAM_COLS      45

typedef enum {
    AUDIO_PREPROCESSING_STATUS_OK = 0,
    AUDIO_PREPROCESSING_STATUS_ERROR_SPECTROGRAM_FULL,
    AUDIO_PREPROCESSING_STATUS_ERROR_OTHER,
} audio_preprocessing_status_t;

audio_preprocessing_status_t audio_preprocessing_init();
audio_preprocessing_status_t audio_preprocessing_process_frame(float32_t * pInSignal);

float32_t *audio_preprocessing_get_spectrogram(void);
uint32_t   audio_preprocessing_get_spectrogram_len(void);
uint32_t   audio_preprocessing_get_spectrogram_filled_cols(void);
audio_preprocessing_status_t   audio_preprocessing_clear_spectrogram(void);
