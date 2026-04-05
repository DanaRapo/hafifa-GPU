#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "../Include/KernelPolyphaseDirect.cuh"
#include "../Include/utils.cuh"
#include "../Include/PolyphaseType.cuh"

/**
 * @brief Executes the WOLA process using cuBLAS matrix multiplication.
 * @param input input data matrix (prepared by prepare_data).
 * @param filter Polyphase filter coefficients matrix.
 * @param output Buffer to store the resulting filtered and summed blocks.
 * @param numBlocks Number of time blocks being processed.
 * @param numChannels Number of channels (subbands).
 * @param decimationFactor Decimation factor (number of samples to skip between blocks).
 * @param olaParam Overlap-add parameter (number of samples to overlap between blocks).
 */
void runWolaDirect(cuComplex* input, cuComplex* filter, cuComplex* output,
     int numBlocks, int numChannels, int decimationFactor, int olaParam);

/**
 * @brief Runs the complete polyphase channelizer pipeline: data preparation, WOLA processing, and output handling.
 * @param cfg Configuration parameters for the polyphase channelizer.
 */
void runCompletePolyphasePipeline(PolyphaseConfig& cfg);