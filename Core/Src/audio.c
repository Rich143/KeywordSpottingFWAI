/**
  ******************************************************************************
  * @file    audio_record.c
  * @author  MCD Application Team
  * @brief   Continuously records mic audio via BSP AUDIO driver using a 2x1s
  *          ping-pong capture buffer. The half/full-transfer callbacks copy
  *          each completed 1-second chunk into a separate 10-second
  *          accumulator buffer. Once 10 seconds have accumulated, capture is
  *          stopped and the whole 10s buffer is sent over UART in polling
  *          (blocking) mode, then capture resumes for the next 10s block.
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

#define ONE_SEC_CHUNK_SIZE  (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * 1U)  /* one 1s sub-chunk */
#define REC_BUFF_SIZE       (2U * ONE_SEC_CHUNK_SIZE)                /* ping-pong capture buffer */

#define ACCUM_SECONDS       10U
#define ACCUM_BUFFER_SIZE   (ACCUM_SECONDS * ONE_SEC_CHUNK_SIZE)     /* 10s accumulator, e.g. ~215 KB -
                                                                          check this fits alongside the
                                                                          rest of your RAM budget (model
                                                                          buffers, etc.) */

#define STREAM_HEADER_MAGIC 0x4B575354U  /* 'KWST' - sent once, describes the stream */
#define CHUNK_HEADER_MAGIC  0x4B575344U  /* 'KWSD' - sent before every chunk */

#define STREAM_UART         (&huart1)    /* CHANGE if this collides with your debug/printf UART */

#define MAX_UART_TX_SIZE    32768U       /* HAL_UART_Transmit's Size param is a uint16_t (max 65535) -
                                             stay comfortably under that per call */

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
static uint8_t         RecordBuff[REC_BUFF_SIZE];    /* 2x1s ping-pong capture buffer */
static uint8_t         AccumBuff[ACCUM_BUFFER_SIZE]; /* 10s accumulator */
static volatile uint32_t AccumChunksFilled = 0;      /* how many 1s sub-chunks are in AccumBuff */
static volatile uint32_t AccumBufferReady  = 0;      /* set (in ISR) once AccumBuff holds a full 10s */
static uint32_t          ChunkSeq          = 0;

/* Private function prototypes -----------------------------------------------*/
static void Record_Init(void);
static void SendStreamHeader(void);
static void SendChunk(uint8_t *payload, uint32_t len);
static void StartCapture(void);
static void AccumulateChunk(uint8_t *src);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Continuously capture audio; every 10 seconds, stop briefly, send
  *         the accumulated buffer over UART, then resume capture.
  * @param  None
  * @retval None
  */
int32_t AudioRecord_demo(void)
{
  printf("\n******AUDIO IN -> UART (10s accumulator) EXAMPLE******\n");

  Record_Init();

  printf("Sending stream header.\n");
  SendStreamHeader();

  StartCapture();

  for (;;)
  {
    if (AccumBufferReady != 0U)
    {
      AccumBufferReady = 0U;
      SendChunk(AccumBuff, ACCUM_BUFFER_SIZE);
      while (1);
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
  * @brief  Kick off continuous (circular) capture into the 2x1s ping-pong buffer.
  * @retval None
  */
static void StartCapture(void)
{
  if (BSP_AUDIO_IN_Record(0, (uint8_t *)RecordBuff, REC_BUFF_SIZE) != BSP_ERROR_NONE)
  {
    printf("AUDIO IN : FAILED.\n");
    Error_Handler();
  }
}

/**
  * @brief  Copy one completed 1-second sub-chunk into the 10s accumulator.
  *         Called from ISR context (half/full-transfer callbacks); the
  *         memcpy is microseconds, well inside the ~1s window before the
  *         DMA needs this half of RecordBuff again, so no separate
  *         double-buffering is needed for AccumBuff itself.
  *         When the accumulator is full, stop capture immediately (still in
  *         ISR context, to close the race against the DMA wrapping around)
  *         and flag the main loop to send it.
  * @param  src Pointer to the completed 1-second region of RecordBuff.
  * @retval None
  */
static void AccumulateChunk(uint8_t *src)
{
  uint8_t *dst = &AccumBuff[AccumChunksFilled * ONE_SEC_CHUNK_SIZE];

  memcpy(dst, src, ONE_SEC_CHUNK_SIZE);
  AccumChunksFilled++;

  if (AccumChunksFilled >= ACCUM_SECONDS)
  {
    BSP_AUDIO_IN_Stop(0);
    AccumChunksFilled = 0U;
    AccumBufferReady  = 1U;
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
  hdr.chunk_size_bytes  = ACCUM_BUFFER_SIZE;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
}

/**
  * @brief  Send one completed 10s chunk (header + payload) over UART, in
  *         polling mode. Capture is stopped for the entire duration of this
  *         call (restarted by the caller afterwards), so there's no time
  *         pressure and no risk of the mic DMA overwriting data mid-send.
  *         HAL_UART_Transmit's Size parameter is a uint16_t, so a payload
  *         over 65535 bytes has to go out as multiple calls - the header is
  *         still sent exactly once, so the receiver sees it as one chunk.
  * @param  payload Pointer to the buffer to send.
  * @param  len     Number of bytes to send.
  * @retval None
  */
static void SendChunk(uint8_t *payload, uint32_t len)
{
  ChunkHeader_t hdr;
  uint32_t      offset = 0U;

  hdr.magic = CHUNK_HEADER_MAGIC;
  hdr.seq   = ChunkSeq++;
  hdr.len   = len;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);

  while (offset < len)
  {
    uint32_t remaining  = len - offset;
    uint16_t send_size  = (remaining > MAX_UART_TX_SIZE) ? (uint16_t)MAX_UART_TX_SIZE : (uint16_t)remaining;

    HAL_UART_Transmit(STREAM_UART, &payload[offset], send_size, HAL_MAX_DELAY);
    offset += send_size;
  }
}

/**
* @brief  Manage the BSP audio in half transfer complete event.
*         First half of RecordBuff (1s sub-chunk 0) is complete - accumulate it.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
  AccumulateChunk(&RecordBuff[0]);
}

/**
* @brief  Manage the BSP audio in transfer complete event.
*         Second half of RecordBuff (1s sub-chunk 1) is complete - accumulate it.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
  AccumulateChunk(&RecordBuff[ONE_SEC_CHUNK_SIZE]);
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
