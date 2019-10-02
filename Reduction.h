#pragma once

#include <ctime>

#define NUM_BLOCKS    64
#define NUM_THREADS   256

void timedReduction(const float *input, float *output, clock_t *timer);

