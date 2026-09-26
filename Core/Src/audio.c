/**
  ******************************************************************************
  * @file    audio_record.c
  * @author  MCD Application Team
  * @brief   Records one buffer-full of mic audio via BSP AUDIO driver, stops
  *          capture, then sends that buffer to a host PC over UART - then
  *          repeats. Capture and transmit never overlap, so UART throughput
  *          doesn't need to keep pace with the mic in real time.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "b_u585i_iot02a_audio.h"

/** @addtogroup STM32U5xx_HAL_Examples
  * @{
  */

/** @addtogroup BSP
  * @{
  */

/* Private define ------------------------------------------------------------*/
#define SAMPLE_RATE_HZ      11025U   /* nominal - matches AUDIO_FREQUENCY_11K request,
                                         still under investigation vs. measured rate */
#define BITS_PER_SAMPLE     16U
#define NUM_CHANNELS        1U
#define BYTES_PER_SAMPLE    (BITS_PER_SAMPLE / 8U)

#define CHUNK_SECONDS       1U
#define BUFFER_SIZE         (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHUNK_SECONDS)  /* one capture buffer */

#define STREAM_HEADER_MAGIC 0x4B575354U  /* 'KWST' - sent once, describes the stream */
#define CHUNK_HEADER_MAGIC  0x4B575344U  /* 'KWSD' - sent before every chunk */

#define STREAM_UART         (&huart1)    /* CHANGE if this collides with your debug/printf UART */

/* Private typedef -------------------------------------------------------------*/
#pragma pack(push, 1)
typedef struct
{
  uint32_t magic;
  uint32_t sample_rate_hz;
  uint16_t bits_per_sample;
  uint16_t num_channels;
  uint32_t chunk_size_bytes;
} StreamHeader_t;

typedef struct
{
  uint32_t magic;
  uint32_t seq;
  uint32_t len;
} ChunkHeader_t;
#pragma pack(pop)

/* Private variables ---------------------------------------------------------*/
static uint8_t         RecordBuff[BUFFER_SIZE];
static __IO uint32_t   BufferReady = 0;   /* set (in ISR) once a full buffer has been
                                              captured and DMA has been stopped */
static uint32_t        ChunkSeq    = 0;

/* Private function prototypes -----------------------------------------------*/
static void Record_Init(void);
static void SendStreamHeader(void);
static void SendChunk(uint8_t *payload, uint32_t len);
static void StartCapture(void);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Repeatedly: capture one buffer, stop, send it over UART, repeat.
  * @param  None
  * @retval None
  */
int32_t AudioRecord_demo(void)
{
  printf("\n******AUDIO IN -> UART (capture-then-send) EXAMPLE******\n");

  Record_Init();

  printf("Sending stream header.\n");
  SendStreamHeader();

  StartCapture();

  for (;;)
  {
    if (BufferReady != 0U)
    {
      BufferReady = 0U;
      SendChunk(RecordBuff, BUFFER_SIZE);
      StartCapture();
    }
  }
}

/**
* @brief  Record initialization
* @param  None
* @retval None
*/
static void Record_Init(void)
{
  BSP_AUDIO_Init_t AudioInit;

  AudioInit.Device        = AUDIO_IN_DEVICE_DIGITAL_MIC1;
  AudioInit.SampleRate    = AUDIO_FREQUENCY_11K;
  AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
  AudioInit.ChannelsNbr   = NUM_CHANNELS;
  AudioInit.Volume        = 100; /* Not used */
  if (BSP_AUDIO_IN_Init(0, &AudioInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }
}

/**
  * @brief  Kick off capture of one full BUFFER_SIZE buffer.
  * @retval None
  */
static void StartCapture(void)
{
  if (BSP_AUDIO_IN_Record(0, (uint8_t *)RecordBuff, BUFFER_SIZE) != BSP_ERROR_NONE)
  {
    printf("AUDIO IN : FAILED.\n");
    Error_Handler();
  }
}

/**
  * @brief  Send the one-time stream header describing sample rate / format.
  *         Sent blocking since it's tiny and only happens once at startup.
  * @retval None
  */
static void SendStreamHeader(void)
{
  StreamHeader_t hdr;

  hdr.magic            = STREAM_HEADER_MAGIC;
  hdr.sample_rate_hz    = SAMPLE_RATE_HZ;
  hdr.bits_per_sample   = BITS_PER_SAMPLE;
  hdr.num_channels      = NUM_CHANNELS;
  hdr.chunk_size_bytes  = BUFFER_SIZE;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
}

/**
  * @brief  Send one completed chunk (header + payload) over UART, blocking.
  *         Capture is stopped for the entire duration of this call (it's
  *         restarted by the caller afterwards), so there's no time pressure
  *         and no risk of the mic DMA overwriting data mid-send.
  * @param  payload Pointer to the buffer to send.
  * @param  len     Number of bytes to send.
  * @retval None
  */
static void SendChunk(uint8_t *payload, uint32_t len)
{
  ChunkHeader_t hdr;

  hdr.magic = CHUNK_HEADER_MAGIC;
  hdr.seq   = ChunkSeq++;
  hdr.len   = len;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
  HAL_UART_Transmit(STREAM_UART, payload, len, HAL_MAX_DELAY);
}

/**
* @brief  Manage the BSP audio in half transfer complete event.
*         Unused in capture-then-send mode (only the full buffer matters).
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
  /* not used */
}

/**
* @brief  Manage the BSP audio in transfer complete event.
*         The buffer is completely full - stop the DMA immediately (from
*         ISR context) before it wraps around and starts overwriting it,
*         then flag the main loop to send it.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
  BSP_AUDIO_IN_Stop(Instance);
  BufferReady = 1U;
}

/**
* @brief  Manages the BSP audio in error event.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance)
{
  Error_Handler();
}

/**
* @}
*/
