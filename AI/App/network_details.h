/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-09-20T10:41:50-0400
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
  .tensors = (const stai_tensor[6]) {
   { .size_bytes = 1350, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 30, 45}}, .scale = {1, (const float[1]){0.017803886905312538}}, .zeropoint = {1, (const int16_t[1]){-35}}, .name = "serving_default_keras_tensor_1050_output" },
   { .size_bytes = 1120, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 7, 10, 16}}, .scale = {1, (const float[1]){0.02584802359342575}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_4_output" },
   { .size_bytes = 960, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 5, 8, 24}}, .scale = {1, (const float[1]){0.01781584694981575}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_6_output" },
   { .size_bytes = 24, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 24}}, .scale = {1, (const float[1]){0.02498709224164486}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_7_output" },
   { .size_bytes = 5, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {1, (const float[1]){0.05236463248729706}}, .zeropoint = {1, (const int16_t[1]){3}}, .name = "gemm_9_output" },
   { .size_bytes = 5, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 5}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_10_output" }
  },
  .nodes = (const stai_node_details[5]){
    {.id = 5, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d_4 */
    {.id = 6, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv2d_6 */
    {.id = 8, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_7 */
    {.id = 9, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* gemm_9 */
    {.id = 10, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} } /* nl_10 */
  },
  .n_nodes = 5
};
#endif

