#pragma once
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include "../Include/KernelCublas.cuh"
#include "../Include/utils.cuh"
#include "../Include/PolyphaseType.cuh"

/**
 * @brief Executes the WOLA process using cuBLAS matrix multiplication.
 * @param handle Valid cuBLAS library handle.
 * @param data_mat Pre-arranged input data matrix (prepared by prepare_data).
 * @param filter_mat Polyphase filter coefficients matrix.
 * @param output Buffer to store the resulting filtered and summed blocks.
 * @param num_blocks Number of time blocks being processed.
 */
void runWola(cublasHandle_t handle, cuComplex* input_stream, cuComplex* filter_mat, 
              cuComplex* output, int num_blocks, int num_channels, int decimation_factor, int ola_param);

 /**
* @brief Executes the complete polyphase filtering pipeline using cuBLAS.
* @param cfg Configuration parameters for the polyphase filter.
*/
void runCompleteCublasPipeline(PolyphaseConfig& cfg);