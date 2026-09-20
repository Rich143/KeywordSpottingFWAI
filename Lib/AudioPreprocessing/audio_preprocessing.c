#include <string.h>
#include <math.h>

#include "feature_extraction.h"
#include "arm_math.h"

#include "audio_preprocessing.h"

#define NFFT             AUDIO_SPECTROGRAM_FRAME_LEN
#define MEL_FMIN_HZ      125
#define MEL_FMAX_HZ      7500

// Log Scale and normalization
#define LOG_EPSILON      1e-6
/// Calculated using tf normalization layer across whole dataset
#define LOG_MEL_MEAN     -5.471536159515381
#define LOG_MEL_STDDEV   5.04770844844

/* Private macro ------------------------------------------------------------*/

/* Private variables --------------------------------------------------------*/
static float32_t
    aSpectrogram[AUDIO_SPECTROGRAM_ROWS * AUDIO_SPECTROGRAM_COLS]; // Stored in
                                                       // column-major format
                                                       // for easier access to
                                                       // a frames worth
                                                       // (single column) of
                                                       // data
static uint32_t SpectrColIndex;
float32_t aWorkingBuffer1[NFFT];

uint32_t melFilterStartIndices[AUDIO_SPECTROGRAM_NMELS];
uint32_t melFilterStopIndices[AUDIO_SPECTROGRAM_NMELS];
float32_t melFilterCoefficients[447];

static arm_rfft_fast_instance_f32 S_Rfft;
static MelFilterTypeDef           S_MelFilter;
static SpectrogramTypeDef         S_Spectr;
static MelSpectrogramTypeDef      S_MelSpectr;

/* Public functions ---------------------------------------------------------*/
audio_preprocessing_status_t audio_preprocessing_init(void) {
    /* Init RFFT */
    arm_rfft_fast_init_512_f32(&S_Rfft);

    /* Init Spectrogram */
    S_Spectr.pRfft    = &S_Rfft;
    S_Spectr.Type     = SPECTRUM_TYPE_POWER;
    S_Spectr.pWindow  = (float32_t *) hannWin_512;
    S_Spectr.SampRate = AUDIO_SAMPLE_RATE_HZ;
    S_Spectr.FrameLen = AUDIO_SPECTROGRAM_FRAME_LEN;
    S_Spectr.FFTLen   = AUDIO_SPECTROGRAM_FRAME_LEN;
    S_Spectr.pScratch = aWorkingBuffer1;

    /* Init Mel filter */
    S_MelFilter.pStartIndices = (uint32_t *) melFilterStartIndices;
    S_MelFilter.pStopIndices  = (uint32_t *) melFilterStopIndices;
    S_MelFilter.pCoefficients = (float32_t *) melFilterCoefficients;
    S_MelFilter.NumMels       = AUDIO_SPECTROGRAM_NMELS;
    S_MelFilter.FFTLen        = AUDIO_SPECTROGRAM_FRAME_LEN;
    S_MelFilter.SampRate      = AUDIO_SAMPLE_RATE_HZ;
    S_MelFilter.FMin          = MEL_FMIN_HZ;
    S_MelFilter.FMax          = MEL_FMAX_HZ;
    S_MelFilter.Formula       = MEL_SLANEY;
    S_MelFilter.Normalize     = 1; // Area normalization enabled
    S_MelFilter.Mel2F         = 0;
    MelFilterbank_Init(&S_MelFilter);


    /* Init MelSpectrogram */
    S_MelSpectr.SpectrogramConf = &S_Spectr;
    S_MelSpectr.MelFilter       = &S_MelFilter;

    SpectrColIndex = 0;

    return AUDIO_PREPROCESSING_STATUS_OK;
}

void log_scale_normalize_column(float32_t *column) {

    for (uint32_t i = 0; i < AUDIO_SPECTROGRAM_ROWS; i++) {
        column[i] = logf(column[i] + LOG_EPSILON);
    }

    // subtract global mean
    arm_offset_f32(column, -LOG_MEL_MEAN, column,
                   AUDIO_SPECTROGRAM_ROWS);
    // divide by global std
    arm_scale_f32(column, 1.0f/LOG_MEL_STDDEV, column,
                  AUDIO_SPECTROGRAM_ROWS);
}

audio_preprocessing_status_t audio_preprocessing_process_frame(float32_t * pInSignal) {
    if (SpectrColIndex >= AUDIO_SPECTROGRAM_COLS) {
        return AUDIO_PREPROCESSING_STATUS_ERROR_SPECTROGRAM_FULL;
    }

    MelSpectrogramColumn(&S_MelSpectr, pInSignal,
                         &aSpectrogram[SpectrColIndex * AUDIO_SPECTROGRAM_ROWS]);

    log_scale_normalize_column(&aSpectrogram[SpectrColIndex * AUDIO_SPECTROGRAM_ROWS]);

    SpectrColIndex++;

    return AUDIO_PREPROCESSING_STATUS_OK;
}

float32_t * audio_preprocessing_get_spectrogram(void) {
    return aSpectrogram;
}

uint32_t audio_preprocessing_get_spectrogram_len(void) {
    return AUDIO_SPECTROGRAM_ROWS * AUDIO_SPECTROGRAM_COLS;
}

uint32_t audio_preprocessing_get_spectrogram_filled_cols(void) {
    return SpectrColIndex;
}

audio_preprocessing_status_t audio_preprocessing_clear_spectrogram(void) {
    SpectrColIndex = 0;

    return AUDIO_PREPROCESSING_STATUS_OK;
}
