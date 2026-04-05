#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * @brief initializes a vector with values from a file
 * @param vec the vector to initialize
 * @param filePath the path to the file containing the values
 */
void initVecFile(std::vector<cuComplex>& vec, const std::string& filePath);

/**
 * @brief initializes a vector with values from a file
 * @param vec the vector to initialize
 * @param filePath the path to the file containing the values
 */
void initRealVecFile(std::vector<float>& vec, const std::string& filePath);

/**
 * @brief writes the content of a vector to a file
 * @param filePath the path to the file to create
 * @param vec the vector to write
 * @param length the length of the vector
 */
void ansToFile(const std::string& filePath, cuComplex* vec, int length);

/**
 * @brief Calculates polyphase parameters (M, R, OLA) based on system requirements.
 * @param fsHz Input sampling rate in Hz.
 * @param channelBw Desired bandwidth per channel in Hz.
 * @param overlapFactor Overlap ratio.
 * @param filterLen Total number of coefficients in the filter.
 * @param inputLen Total number of samples in the input signal.
 * @param[out] numBlocks Total time blocks to process.
 * @param[out] numChannels Number of FFT channels (M).
 * @param[out] decimationFactor Step size between blocks (R).
 * @param[out] olaParam Number of polyphase taps per channel (OLA).
 */
void polyphaseParams(float fsHz, float channelBw, float overlapFactor, int filterLen, int inputLen,
     int& numBlocks, int& numChannels, int& decimationFactor, int& olaParam);

/**
 * @brief Checks the result of a CUDA API call and throws an exception if an error occurred.
 * @param result The result of the CUDA API call to check.
 * @throws std::runtime_error if the CUDA API call returned an error.
 */
void checkCuda(cudaError_t result);