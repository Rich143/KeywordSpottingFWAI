/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-16T14:45:04-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x64a5c568febc4e7ea000a6b8824e249c"
#define _STAI_NETWORK_DATETIME            "2026-09-16T14:45:04-0400"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 1),
  .tool_version = STAI_INIT_VERSION(4, 0, 1),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 30, 45),
      STAI_DECLARE_ARRAY(float, 1, 0.017803886905312538f),
      STAI_DECLARE_ARRAY(int16_t, 1, -35)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 5),
      STAI_DECLARE_ARRAY(float, 1, 0.07681304216384888f),
      STAI_DECLARE_ARRAY(int16_t, 1, -51)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 16880),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 42740),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      NULL
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input_layer0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017803886905312538f),
    AI_PACK_INTQ_ZP(-35)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025523865595459938f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007253623567521572f, 0.01603534258902073f, 0.007109351921826601f, 0.006694843526929617f, 0.010808892548084259f, 0.007692186161875725f, 0.007329232059419155f, 0.002347594825550914f, 0.0040529570542275906f, 0.011048841290175915f, 0.006670891307294369f, 0.008352474309504032f, 0.010441496968269348f, 0.003639791626483202f, 0.00755211990326643f, 0.003394326427951455f, 0.00326726702041924f, 0.0038293839897960424f, 0.0030755200423300266f, 0.011348871514201164f, 0.0047465041279792786f, 0.0025450834073126316f, 0.007281993515789509f, 0.011687928810715675f, 0.003515929915010929f, 0.00288136163726449f, 0.003653820138424635f, 0.006870707497000694f, 0.0044186851009726524f, 0.00802764855325222f, 0.0023401950020343065f, 0.007375258021056652f, 0.004207877442240715f, 0.0032766996882855892f, 0.005426168907433748f, 0.00336509570479393f, 0.004928105976432562f, 0.0067620608024299145f, 0.0040633054450154305f, 0.00518966605886817f, 0.009552989155054092f, 0.002939694793894887f, 0.00233836960978806f, 0.0031499594915658236f, 0.009240016341209412f, 0.003384625306352973f, 0.006874005775898695f, 0.0027980946470052004f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_scratch1_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025523865595459938f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_pad_before_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02209790237247944f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03820197284221649f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002296548569574952f, 0.0023919781669974327f, 0.002211221493780613f, 0.002126139821484685f, 0.00215897080488503f, 0.0016865546349436045f, 0.002084944862872362f, 0.0021321263629943132f, 0.0020975214429199696f, 0.001967113697901368f, 0.0010671727359294891f, 0.0009472445235587656f, 0.0011160445865243673f, 0.002263051690533757f, 0.0018130181124433875f, 0.0009851752547547221f, 0.001840319368056953f, 0.002391105517745018f, 0.0009820600971579552f, 0.0012968763476237655f, 0.001705754199065268f, 0.002717970870435238f, 0.0012111521791666746f, 0.0015701574739068747f, 0.0017787624383345246f, 0.0015748817240819335f, 0.0022300281561911106f, 0.0007196553633548319f, 0.0015022449661046267f, 0.0017984173027798533f, 0.002534746890887618f, 0.0027531678788363934f, 0.0021275444887578487f, 0.0019148149294778705f, 0.002533277263864875f, 0.0019554344471544027f, 0.0010072145378217101f, 0.0019518053159117699f, 0.001261487021110952f, 0.002611292526125908f, 0.0009127817465923727f, 0.002368706976994872f, 0.0025751490611582994f, 0.001348573830910027f, 0.0019222571281716228f, 0.0018911466468125582f, 0.0012578063178807497f, 0.0007924525416456163f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_scratch1_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03820197284221649f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07681304216384888f),
    AI_PACK_INTQ_ZP(-51)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_9_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 5,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027532116509974003f, 0.0025636241771280766f, 0.0025557863991707563f, 0.0025402510073035955f, 0.0027705200482159853f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_layer0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1350, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3696, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1572, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8640, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5616, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 48, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20736, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7520, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3696, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 240, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 5, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_9_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 73, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 11, 7), AI_STRIDE_INIT(4, 1, 1, 48, 528),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1572, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1572, 1572),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch1, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 45, 4), AI_STRIDE_INIT(4, 1, 1, 48, 2160),
  1, &conv2d_4_scratch1_array, &conv2d_4_scratch1_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 48), AI_STRIDE_INIT(4, 1, 1, 48, 144),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer0_output0, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 45, 30), AI_STRIDE_INIT(4, 1, 1, 1, 45),
  1, &serving_default_input_layer0_output_array, &serving_default_input_layer0_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_pad_before_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 13, 9), AI_STRIDE_INIT(4, 1, 1, 48, 624),
  1, &conv2d_7_pad_before_output_array, &conv2d_7_pad_before_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 7520, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7520, 7520),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch1, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 11, 7), AI_STRIDE_INIT(4, 1, 1, 48, 528),
  1, &conv2d_7_scratch1_array, &conv2d_7_scratch1_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 48, 3, 3, 48), AI_STRIDE_INIT(4, 1, 48, 2304, 6912),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &gemm_9_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &gemm_9_output_array, &gemm_9_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 73, 1, 1), AI_STRIDE_INIT(4, 2, 2, 146, 146),
  1, &gemm_9_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_9_weights, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 48, 5, 1, 1), AI_STRIDE_INIT(4, 1, 48, 240, 240),
  1, &gemm_9_weights_array, &gemm_9_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_layer0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_4_scratch0, &conv2d_4_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 5,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &conv2d_4_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(4, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_7_scratch0, &conv2d_7_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 8,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_7_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(11, 7), 
  .pool_stride = AI_SHAPE_2D_INIT(11, 7), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_9_weights, &gemm_9_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_9_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_9_layer, 9,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_9_chain,
  NULL, &gemm_9_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_conv2d_sssa8_ch_nl_pool_conv2d_4(_stai_network_context* net_ctx)
{
  serving_default_input_layer0_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  serving_default_input_layer0_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  conv2d_4_weights_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  conv2d_4_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  conv2d_4_bias_array.data = AI_PTR(net_ctx->_weights[0] + 432);
  conv2d_4_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 432);
  conv2d_4_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 12336);
  conv2d_4_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 12336);
  conv2d_4_scratch1_array.data = AI_PTR(net_ctx->_activations[0] + 3696);
  conv2d_4_scratch1_array.data_start = AI_PTR(net_ctx->_activations[0] + 3696);
  conv2d_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, { serving_default_input_layer0_output0.data->data});
  forward_conv2d_sssa8_ch_nl_pool(&conv2d_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, { conv2d_4_output.data->data});
}
void forward_lite_conv2d_deep_3x3_sssa8_ch_nl_pool_conv2d_7(_stai_network_context* net_ctx)
{
  conv2d_7_pad_before_output_array.data = AI_PTR(net_ctx->_activations[0] + 3696);
  conv2d_7_pad_before_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3696);
  conv2d_7_weights_array.data = AI_PTR(net_ctx->_weights[0] + 21552);
  conv2d_7_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 21552);
  conv2d_7_bias_array.data = AI_PTR(net_ctx->_weights[0] + 42288);
  conv2d_7_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 42288);
  conv2d_7_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 9312);
  conv2d_7_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 9312);
  conv2d_7_scratch1_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_7_scratch1_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 16832);
  conv2d_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16832);
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, { conv2d_7_pad_before_output.data->data});
   forward_conv2d_deep_3x3_sssa8_ch_nl_pool(&conv2d_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, { conv2d_7_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_9(_stai_network_context* net_ctx)
{
  conv2d_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 16832);
  conv2d_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16832);
  gemm_9_weights_array.data = AI_PTR(net_ctx->_weights[0] + 42480);
  gemm_9_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 42480);
  gemm_9_bias_array.data = AI_PTR(net_ctx->_weights[0] + 42720);
  gemm_9_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 42720);
  gemm_9_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_9_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_9_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  gemm_9_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, { conv2d_7_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, { gemm_9_output.data->data});
}

/*****************************************************************************/



static const ai_i8 conv2d_6_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_6_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_6_pad_before_t_in_0_shape_h_const_u32 = 7;

static const ai_u16 conv2d_6_t_in_0_shape_w_const_u16 = 13;
static const ai_u16 conv2d_6_t_in_0_shape_h_const_u16 = 9;
static const ai_u16 conv2d_6_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_6_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_6_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_6_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_6_t_in_0_fmt_scale_const_f32 = 0.025523865595459938f;
static const ai_float conv2d_6_t_out_0_fmt_scale_const_f32 = 0.02209790237247944f;
static const ai_float conv2d_6_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.000723467324860394f, 0.0015620074700564146f, 0.0011890146415680647f, 0.0005883207777515054f, 0.0013511184370145202f, 0.0011684659402817488f, 0.0011552966898307204f, 0.0008677755831740797f, 0.0010460260091349483f, 0.0011756006861105561f, 0.000984631828032434f, 0.0009032333036884665f, 0.0015959326410666108f, 0.0010758494026958942f, 0.0005333435256034136f, 0.0010223835706710815f, 0.0008332178113050759f, 0.0016892824787646532f, 0.0014165014727041125f, 0.0011267283698543906f, 0.0010735660325735807f, 0.0012975342106074095f, 0.0012168860994279385f, 0.0015038764104247093f, 0.0006630514981225133f, 0.0007612396730110049f, 0.0015390381449833512f, 0.00039767255657352507f, 0.001522549893707037f, 0.0007854662253521383f, 0.0010044778464362025f, 0.0012280060909688473f, 0.0015423601726070046f, 0.0009611774003133178f, 0.001459808787330985f, 0.001220445497892797f, 0.0009633026202209294f, 0.0012317767832428217f, 0.0017719271127134562f, 0.0006725860293954611f, 0.001145059010013938f, 0.0014557484537363052f, 0.0011742461938410997f, 0.0007968682330101728f, 0.0008170779328793287f, 0.0012870775535702705f, 0.0012867036275565624f, 0.0009956738213077188f);
static const ai_layer_format_type conv2d_6_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_6_t_out_0_shape_w_const_u16 = 11;
static const ai_u16 conv2d_6_t_out_0_shape_h_const_u16 = 7;

static const ai_i8 conv2d_7_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_7_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_7_pad_before_t_in_0_shape_h_const_u32 = 7;


STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conv2d_4 */
  {
    
  forward_lite_conv2d_sssa8_ch_nl_pool_conv2d_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END conv2d_4 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_6_pad_before */
  {
      const ai_ptr conv2d_6_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
    ai_ptr conv2d_6_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 3696);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(6, 1, {(stai_ptr) conv2d_6_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_6_pad_before_t_in_0_ptr_const_ptr, conv2d_6_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_6_pad_before_v_pad_constant_value_const_s8), conv2d_6_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_6_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(528), (ai_i32)(624), (ai_i32)(624), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(6, 1, {(stai_ptr) conv2d_6_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_6_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_6 */
  {
      const ai_i8* conv2d_6_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 3696);
    const ai_i8* conv2d_6_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 624);
    const ai_i32* conv2d_6_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 21360);
    ai_i8* conv2d_6_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_6_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 9312);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(6, 1, {(stai_ptr) conv2d_6_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_3x3_sssa8_ch(conv2d_6_t_in_0_ptr_const_s8, conv2d_6_t_in_0_shape_w_const_u16, conv2d_6_t_in_0_shape_h_const_u16, conv2d_6_t_in_0_shape_ch_const_u16, conv2d_6_t_weight_0_ptr_const_s8, conv2d_6_t_out_0_shape_ch_const_u16, conv2d_6_t_weight_1_ptr_const_s32, conv2d_6_t_in_0_fmt_zero_const_s8, conv2d_6_t_out_0_fmt_zero_const_s8, conv2d_6_t_in_0_fmt_scale_const_f32, conv2d_6_t_out_0_fmt_scale_const_f32, conv2d_6_t_weight_0_fmt_scale_const_f32, conv2d_6_l_out_ch_format_const_layer_format_type, conv2d_6_t_out_0_ptr_s8, conv2d_6_t_out_0_shape_w_const_u16, conv2d_6_t_out_0_shape_h_const_u16, 1, 7520, conv2d_6_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(6, 1, {(stai_ptr) conv2d_6_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_6 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_7_pad_before */
  {
      const ai_ptr conv2d_7_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
    ai_ptr conv2d_7_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 3696);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(7, 1, {(stai_ptr) conv2d_7_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_7_pad_before_t_in_0_ptr_const_ptr, conv2d_7_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_7_pad_before_v_pad_constant_value_const_s8), conv2d_7_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_7_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(528), (ai_i32)(624), (ai_i32)(624), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) conv2d_7_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_7_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_7 */
  {
    
  forward_lite_conv2d_deep_3x3_sssa8_ch_nl_pool_conv2d_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END conv2d_7 */
  /* LITE_KERNEL_SECTION BEGIN gemm_9 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_9 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 13908;

  net_ctx->_outputs[0] = activations[0] + 148;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

