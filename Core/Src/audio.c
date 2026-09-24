/**
  ******************************************************************************
  * @file    audio_record.c
  * @author  MCD Application Team
  * @brief   Records mic audio via BSP AUDIO driver and streams it to a host
  *          PC over UART in 1-second chunks, using DMA half/full-transfer
  *          interrupts as the ping-pong trigger.
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
#define SAMPLE_RATE_HZ      11025U   /* matches AUDIO_FREQUENCY_11K */
#define BITS_PER_SAMPLE     16U
#define NUM_CHANNELS        1U
#define BYTES_PER_SAMPLE    (BITS_PER_SAMPLE / 8U)

#define CHUNK_SECONDS       1U
#define CHUNK_SIZE          (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHUNK_SECONDS)  /* 22050 bytes */
#define REC_BUFF_SIZE       (2U * CHUNK_SIZE)  /* circular buffer = two chunks (ping-pong) */

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
static uint8_t         RecordBuff[REC_BUFF_SIZE];
static __IO uint32_t   RecHalfBuffCplt   = 0;
static __IO uint32_t   RecBuffCplt       = 0;
static __IO uint32_t   UartTxBusy        = 0;
static uint32_t        ChunkSeq          = 0;
static uint32_t        DroppedChunks     = 0;

/*static volatile uint32_t LastCallbackTick = 0;*/
/*static volatile uint32_t DeltasMs         = 0;*/
/*static volatile uint32_t NumDeltas        = 0;*/


/* Private function prototypes -----------------------------------------------*/
static void Record_Init(void);
static void SendStreamHeader(void);
static void SendChunk(uint8_t *payload);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Record mic audio and stream it over UART in 1-second chunks.
  * @param  None
  * @retval None
  */
int32_t AudioRecord_demo(void)
{
  printf("\n******AUDIO IN -> UART STREAM EXAMPLE******\n");

  Record_Init();

  printf("Sending stream header.\n");
  SendStreamHeader();

  printf("Recording + streaming started (Ctrl-C / reset to stop)..\n");
  if (BSP_AUDIO_IN_Record(0, (uint8_t *)RecordBuff, REC_BUFF_SIZE) != BSP_ERROR_NONE)
  {
    printf("AUDIO IN : FAILED.\n");
    Error_Handler();
  }

  for (;;)
  {
    if (RecHalfBuffCplt != 0U)
    {
      RecHalfBuffCplt = 0U;
      //printf("Sending half buffer.\n");
      SendChunk(&RecordBuff[0]);
    }

    if (RecBuffCplt != 0U)
    {
      RecBuffCplt = 0U;
      //printf("Sending full buffer.\n");
      SendChunk(&RecordBuff[CHUNK_SIZE]);
    }

    /*if (NumDeltas % 20 == 0U)*/
    /*{*/
      /*printf("Delta avg: %d ms\n", DeltasMs / NumDeltas);*/
    /*}*/
  }

  /* Unreachable with the current infinite loop; left in for reference if you
   * add a stop condition (button press, host command, etc.) above. */
  if (BSP_AUDIO_IN_Stop(0) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  if (BSP_AUDIO_IN_DeInit(0) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  RecBuffCplt     = 0;
  RecHalfBuffCplt = 0;
  return 0;
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
  hdr.chunk_size_bytes  = CHUNK_SIZE;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);
}

/**
  * @brief  Send one completed chunk (header + payload) over UART.
  *         The header is sent blocking (12 bytes, negligible time); the
  *         payload is sent via UART DMA so the CPU is free while it goes out.
  *         If the previous chunk's DMA transmit hasn't finished yet, this
  *         chunk is dropped rather than corrupting the in-flight transfer -
  *         at 921600 baud a 1s/22050-byte chunk only takes ~0.24s to send,
  *         so under normal operation this should never trigger.
  * @param  payload Pointer to the CHUNK_SIZE-byte region of RecordBuff that
  *                  the DMA has just finished filling.
  * @retval None
  */
static void SendChunk(uint8_t *payload)
{
  ChunkHeader_t hdr;

  if (UartTxBusy != 0U)
  {
    DroppedChunks++;
    return;
  }

  hdr.magic = CHUNK_HEADER_MAGIC;
  hdr.seq   = ChunkSeq++;
  hdr.len   = CHUNK_SIZE;

  HAL_UART_Transmit(STREAM_UART, (uint8_t *)&hdr, sizeof(hdr), HAL_MAX_DELAY);

  UartTxBusy = 1U;
  if (HAL_UART_Transmit_DMA(STREAM_UART, payload, CHUNK_SIZE) != HAL_OK)
  {
    UartTxBusy = 0U;
    DroppedChunks++;
  }
}

/**
* @brief  Manage the BSP audio in half transfer complete event.
*         First half of RecordBuff (chunk 0) is now safe to send.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
  /*uint32_t now = HAL_GetTick();*/
  /*uint32_t LastDeltaMs = now - LastCallbackTick;*/
  /*LastCallbackTick = now;*/

  /*DeltasMs += LastDeltaMs;*/
  /*NumDeltas++;*/

  RecHalfBuffCplt++;
}

/**
* @brief  Manage the BSP audio in transfer complete event.
*         Second half of RecordBuff (chunk 1) is now safe to send.
* @param  Instance Audio in instance.
* @retval None.
*/
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
  /*uint32_t now = HAL_GetTick();*/
  /*uint32_t LastDeltaMs = now - LastCallbackTick;*/
  /*LastCallbackTick = now;*/

  /*DeltasMs += LastDeltaMs;*/
  /*NumDeltas++;*/

  RecBuffCplt++;
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
  * @brief  UART Tx DMA complete callback - marks the link free for the next chunk.
  * @param  huart UART handle.
  * @retval None
  */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == STREAM_UART->Instance)
  {
    UartTxBusy = 0U;
  }
}

/**
* @}
*/
