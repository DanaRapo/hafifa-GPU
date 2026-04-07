#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

/**
 * @brief Rearranges input stream data into a blocked matrix format for cuBLAS processing.
 * @param inputStream Pointer to the raw input signal buffer.
 * @param dataMat Pointer to the output matrix buffer organized by OLA taps.
 * @param numChannels Number of channels (M) in the filter bank.
 * @param decimationFactor The hop size (R) between consecutive blocks.
 * @param numBlocks Total number of time blocks to process in this batch.
 * @param olaParam Number of OLA taps (L) used in the polyphase filtering.
 */
__global__ void PrepareDataCublas(cuComplex* inputStream, cuComplex* dataMat,
     int numChannels, int decimationFactor, int numBlocks, int olaParam);
