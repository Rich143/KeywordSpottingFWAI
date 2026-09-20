#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdint.h>
#include "arm_math_types.h"

void check_signal(const float32_t *actual, const float32_t *expected,
                  uint32_t len);
void check_signal_close(const float32_t *actual, const float32_t *expected,
                        uint32_t len, float32_t tolerance);

#endif
