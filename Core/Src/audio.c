/**
  ******************************************************************************
  * @file    audio_record.c
  * @author  MCD Application Team
  * @brief   Continuously records mic audio via BSP AUDIO driver using a 2x1s
  *          ping-pong capture buffer. The half/full-transfer callbacks only
  *          flag which 1-second half is complete; the main loop then sends
  *          that half straight out of RecordBuff over UART in polling
  *          (blocking) mode while the DMA fills the other half. Capture is
  *          never stopped.
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
#define SAMPLE_RATE_HZ      AUDIO_FREQUENCY_16K
#define BITS_PER_SAMPLE     16U
#define NUM_CHANNELS        1U
#define BYTES_PER_SAMPLE    (BITS_PER_SAMPLE / 8U)

#define ONE_SEC_CHUNK_SIZE  (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * NUM_CHANNELS)  /* one 1s chunk */
#define REC_BUFF_SIZE       (2U * ONE_SEC_CHUNK_SIZE)                          /* ping-pong capture buffer */

#define STREAM_HEADER_MAGIC 0x4B575354U  /* 'KWST' - sent once, describes the stream */
#define CHUNK_HEADER_MAGIC  0x4B575344U  /* 'KWSD' - sent before every chunk */

#define STREAM_UART         (&huart1)    /* CHANGE if this collides with your debug/printf UART */

#define DISCARD_CHUNKS      8U           /* number of 1s chunks to drop at startup, in case of
                                             startup corruption - tune as needed */

/* Size limits (checked at compile time):
   - BSP_AUDIO_IN_Record() programs the whole buffer as one GPDMA block, and
     the block size field (CBR1.BNDT) is 16 bits, in bytes -> REC_BUFF_SIZE <= 65535.
   - HAL_UART_Transmit()'s Size is a uint16_t -> each chunk is sent in one call.
   - The DMA half-transfer event splits the buffer in two, so each half must
     hold a whole number of samples. */
_Static_assert(REC_BUFF_SIZE <= 0xFFFFU, "REC_BUFF_SIZE exceeds the 16-bit GPDMA block size");
_Static_assert(ONE_SEC_CHUNK_SIZE <= 0xFFFFU, "Chunk exceeds HAL_UART_Transmit's uint16_t Size");
_Static_assert((ONE_SEC_CHUNK_SIZE % BYTES_PER_SAMPLE) == 0U, "Chunk must hold whole samples");

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
static uint8_t           RecordBuff[REC_BUFF_SIZE];  /* 2x1s ping-pong capture buffer */
static volatile uint32_t ChunkReadyCount = 0;        /* incremented (in ISR) per completed 1s half */
static volatile uint8_t *ChunkReadyPtr   = NULL;     /* half of RecordBuff that most recently completed */
static volatile uint32_t DiscardChunksRemaining = DISCARD_CHUNKS; /* counts down to 0 at startup */

/* Private function prototypes -----------------------------------------------*/
static void Record_Init(void);
static void SendStreamHeader(void);
static void SendChunk(const uint8_t *payload, uint32_t seq, uint32_t len);
static void StartCapture(void);
static void ChunkComplete(uint8_t *src);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Continuously capture audio and send each 1s chunk over UART as
  *         soon as its half of RecordBuff is complete.
  *         The chunk is sent directly from RecordBuff: the DMA won't write
  *         that half again until it has filled the other half (1s later), and
  *         the send takes ~240 ms at 921600 baud. If the main loop falls
  *         behind, missed chunks show up as gaps in the sequence number, and
  *         a chunk that may have been overwritten mid-send is reported.
  * @param  None
  * @retval None
  */
int32_t AudioRecord_demo(void)
{
  uint32_t sent_count = 0U;  /* value of ChunkReadyCount for the last chunk sent */

  printf("\n******AUDIO IN -> UART (1s chunk stream) EXAMPLE******\n");

  Record_Init();

  printf("Sending stream header.\n");
  SendStreamHeader();

  StartCapture();

  for (;;)
  {
    uint32_t       ready_count;
    const uint8_t *chunk;

    /* Snapshot count and pointer together so the ISR can't update one
       between the two reads */
    __disable_irq();
    ready_count = ChunkReadyCount;
    chunk       = (const uint8_t *)ChunkReadyPtr;
    __enable_irq();

    if (ready_count != sent_count)
    {
      if ((ready_count - sent_count) > 1U)
      {
        printf("[audio] missed %lu chunk(s)\n", (unsigned long)(ready_count - sent_count - 1U));
      }
      sent_count = ready_count;

      /* seq is the capture count (from 0), so the host sees missed chunks as gaps */
      SendChunk(chunk, ready_count - 1U, ONE_SEC_CHUNK_SIZE);

      if (ChunkReadyCount != ready_count)
      {
        /* The other half finished while we were sending, so the DMA has
           started overwriting the half we just sent */
        printf("[audio] chunk %lu may be corrupt (send overran)\n", (unsigned long)(ready_count - 1U));
      }
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
  AudioInit.SampleRate    = SAMPLE_RATE_HZ;
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
  * @brief  Flag one completed 1-second half of RecordBuff to the main loop.
  *         The first DISCARD_CHUNKS chunks after startup are dropped instead
  *         (not counted), in case the very first bit of captured audio is
  *         corrupted.
  *         Called from ISR context (half/full-transfer callbacks), so it only
  *         records which half is ready; the UART send happens in the main loop.
  * @param  src Pointer to the completed 1-second region of RecordBuff.
  * @retval None
  */
static void ChunkComplete(uint8_t *src)
{
  if (DiscardChunksRemaining > 0U)
  {
    DiscardChunksRemaining--;
    return;
  }

  ChunkReadyPtr = src;
  ChunkReadyCount++;
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
  hdr.chunk_size_bytes  = ONE_SEC_CHUNK_SIZE;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
}

/**
  * @brief  Send one 1s chunk (header + payload) over UART, in polling mode.
  *         At 921600 baud 8N1 (92160 bytes/s) the 12-byte header plus
  *         22050-byte payload takes ~240 ms, well inside the 1s before the
  *         DMA comes back to this half of RecordBuff.
  *         Called from the main loop only - never from the DMA callbacks.
  * @param  payload Pointer to the chunk to send (a half of RecordBuff).
  * @param  seq     Chunk sequence number.
  * @param  len     Number of bytes to send (<= 65535, see _Static_assert).
  * @retval None
  */
static void SendChunk(const uint8_t *payload, uint32_t seq, uint32_t len)
{
  ChunkHeader_t hdr;

  hdr.magic = CHUNK_HEADER_MAGIC;
  hdr.seq   = seq;
  hdr.len   = len;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
  HAL_UART_Transmit(STREAM_UART, (uint8_t *)payload, (uint16_t)len, HAL_MAX_DELAY);
}

/**
* @brief  Manage the BSP audio in half transfer complete event.
*         First half of RecordBuff (1s chunk 0) is complete - flag it.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
  ChunkComplete(&RecordBuff[0]);
}

/**
* @brief  Manage the BSP audio in transfer complete event.
*         Second half of RecordBuff (1s chunk 1) is complete - flag it.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
  ChunkComplete(&RecordBuff[ONE_SEC_CHUNK_SIZE]);
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
