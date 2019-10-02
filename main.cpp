#include <cstdio>
#include <cuda_runtime.h>
#include <ctime>
#include "Reduction.h"

int main(int argc, char **argv) {
    printf("CUDA Clock sample\n");

    // This will pick the best possible CUDA capable device
    cudaSetDevice(0);

    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float) i;
    }

    cudaMalloc((void **) &dinput, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void **) &doutput, sizeof(float) * NUM_BLOCKS);
    cudaMalloc((void **) &dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);

    cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice);

    timedReduction(dinput, doutput, dtimer);

    cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double avgElapsedClocks = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double) (timer[i + NUM_BLOCKS] - timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return 0;
}
