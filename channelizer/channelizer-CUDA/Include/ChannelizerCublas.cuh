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

class ChannelizerCublas{
public:

    ChannelizerCublas();
    
    ~ChannelizerCublas();

    /**
     * @brief Factory method to create and initialize a ChannelizerCublas instance based on the provided configuration.
     * @param cfg Configuration parameters for the polyphase filter.
     * @return A unique pointer to the initialized ChannelizerCublas instance.
     */
    static std::unique_ptr<ChannelizerCublas> createAndLoad(PolyphaseConfig& cfg);
    
    /**
    * @brief Executes the complete polyphase filtering pipeline using cuBLAS.
    * @param cfg Configuration parameters for the polyphase filter, including paths to input, filter
    */
    void runCompleteCublasPipeline(const PolyphaseConfig& cfg);

private:
    /**
     * @brief Private constructor to initialize the ChannelizerCublas instance with the given configuration and allocate necessary resources.
     * @param cfg Configuration parameters for the polyphase filter.
     * @param inputSize Size of the input data buffer to allocate on the device.
     * @param filterSize Size of the filter coefficients buffer to allocate on the device.
     */
    ChannelizerCublas(const PolyphaseConfig& cfg, size_t inputSize, size_t filterSize);
    cublasHandle_t _handle;
    cuComplex *_dInput, *_dFilter, *_dOutput, *_dDataMat, *_dPhaseRes;

    size_t currentInputSize;
    size_t currentMatSize;
    std::vector<cuComplex> m_hData;
    std::vector<cuComplex> m_hFilter;
    size_t m_outSize;

    /**
     * @brief Executes the WOLA process using cuBLAS matrix multiplication.
     * @param handle Valid cuBLAS library handle.
     * @param inputStream Pre-arranged input data matrix (prepared by prepare_data).
     * @param filterMat Polyphase filter coefficients matrix.
     * @param output Buffer to store the resulting filtered and summed blocks.
     * @param dataMat Temporary buffer for the rearranged input data for each OLA block.
     * @param phaseRes Temporary buffer to store the results of the matrix multiplication for each phase.
     * @param numBlocks Number of time blocks being processed.
     * @param numChannels Number of channels (filter coefficients) in the polyphase filter.
     * @param decimationFactor Decimation factor for the WOLA process.
     * @param olaParam Number of overlapping blocks (OLA parameter).
     */
    void runWola(cublasHandle_t handle, cuComplex* inputStream, cuComplex* filterMat, 
                cuComplex* output, cuComplex* dataMat, cuComplex* phaseRes,
                int numBlocks, int numChannels, int decimationFactor, int olaParam);
};

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