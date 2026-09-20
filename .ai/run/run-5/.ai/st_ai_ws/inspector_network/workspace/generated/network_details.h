/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-09-16T14:45:04-0400
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[7]) {
   { .size_bytes = 1350, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 45}}, .scale = {1, (const float[1]){0.017803886905312538}}, .zeropoint = {1, (const int16_t[1]){-35}}, .name = "serving_default_input_layer0_output" },
   { .size_bytes = 3696, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 7, 11, 48}}, .scale = {1, (const float[1]){0.025523865595459938}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_4_output" },
   { .size_bytes = 5616, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 9, 13, 48}}, .scale = {1, (const float[1]){0.025523865595459938}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_6_pad_before_output" },
   { .size_bytes = 3696, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 7, 11, 48}}, .scale = {1, (const float[1]){0.02209790237247944}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_6_output" },
   { .size_bytes = 5616, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 9, 13, 48}}, .scale = {1, (const float[1]){0.02209790237247944}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_7_pad_before_output" },
   { .size_bytes = 48, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 48}}, .scale = {1, (const float[1]){0.03820197284221649}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_7_output" },
   { .size_bytes = 5, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {1, (const float[1]){0.07681304216384888}}, .zeropoint = {1, (const int16_t[1]){-51}}, .name = "gemm_9_output" }
  },
  .nodes = (const stai_node_details[6]){
    {.id = 5, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d_4 */
    {.id = 6, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv2d_6_pad_before */
    {.id = 6, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_6 */
    {.id = 7, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* conv2d_7_pad_before */
    {.id = 8, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* conv2d_7 */
    {.id = 9, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} } /* gemm_9 */
  },
  .n_nodes = 6
};
#endif

