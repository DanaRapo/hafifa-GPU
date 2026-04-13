#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cublas_v2.h>
#include <memory>
#include "../Include/utils.cuh"
#include "../Include/PolyphaseType.cuh"

class ChannelizerDirect {
public:
    ~ChannelizerDirect();
    ChannelizerDirect();

    static std::unique_ptr<ChannelizerDirect> createAndLoad(PolyphaseConfig& cfg);

    /**
     * @brief Runs the complete polyphase channelizer pipeline: data preparation, WOLA processing, and output handling.
     * @param cfg Configuration parameters for the polyphase channelizer.
     */
    void runCompleteDirectPipeline(PolyphaseConfig& cfg);

private:
    cuComplex *_dInput, *_dFilter, *_dOutput;
    std::vector<cuComplex> m_hData;
    std::vector<cuComplex> m_hFilter;
    size_t m_outSize;
    int m_sharedMemSize;

    ChannelizerDirect(const PolyphaseConfig& cfg, size_t inputSize, size_t filterSize);

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

};

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
__global__ void oldwolaDirect( const cuComplex* __restrict__ input, const cuComplex* __restrict__ filter,
    cuComplex* output, int olaParam, int numChannels, int decimationFactor, int numBlocks);