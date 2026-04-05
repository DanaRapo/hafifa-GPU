#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cublas_v2.h>

/**
 * @brief CUDA kernel for performing the WOLA polyphase filtering.
 * @param input Pointer to the raw input signal buffer. using  __restrict__  for the compiler to load data to read only cache, which can improve performance when multiple threads access the same data.
 * @param filter Pointer to the filter coefficients buffer.
 * @param output Pointer to the output matrix buffer organized by OLA taps.
 * @param olaParam OLA parameter for the filtering operation.
 * @param numChannels Number of channels (M) in the filter bank.
 * @param decimationFactor The hop size (R) between consecutive blocks.
 * @param numBlocks Total number of time blocks to process in this batch.
 */
__global__ void wolaDirect( const cuComplex* __restrict__ input, const cuComplex* __restrict__ filter,
    cuComplex* output, int olaParam, int numChannels, int decimationFactor, int numBlocks);