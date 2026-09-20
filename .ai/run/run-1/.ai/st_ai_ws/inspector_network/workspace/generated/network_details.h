/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-09-09T11:11:30-0400
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
   { .size_bytes = 5400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 45}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_0_output" },
   { .size_bytes = 259200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 45, 48}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_output" },
   { .size_bytes = 259200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 30, 45, 48}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "activation_output" },
   { .size_bytes = 14784, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 7, 11, 48}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "max_pooling2d_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 48}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_dense_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 48}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_output" },
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "dense_1_output" }
  },
  .nodes = (const stai_node_details[6]){
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* activation */
    {.id = 4, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* max_pooling2d */
    {.id = 6, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* dense_dense */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* dense */
    {.id = 7, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} } /* dense_1 */
  },
  .n_nodes = 6
};
#endif

