#include "unity.h"
#include "test_helpers.h"

void check_signal(const float32_t *actual, const float32_t *expected,
                  uint32_t len)
{
    for (uint32_t i = 0; i < len; ++i) {
        char message[128];

        sprintf(message, "Value mismatch at index %u", i);
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.001f, expected[i], actual[i], message);
    }
}

void check_signal_close(const float32_t *actual, const float32_t *expected,
                        uint32_t len, float32_t tolerance)
{
    TEST_ASSERT_FLOAT_ARRAY_WITHIN_MESSAGE(tolerance, expected, actual,
                                                len, "Value mismatch");
}


