#pragma once

#include <stdint.h>

uint32_t __get_PRIMASK(void);

void __disable_irq(void);

void __DMB(void);

void __set_PRIMASK(uint32_t primask);
