#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
/**
 * @brief sums to vectors
 * @param vec1 first vector to add
 * @param vec2 second vector to add
 * @param ansVec the vector that saves the answers
 * @param vectorLen the vectors length
 */
__global__ void vecAdd(float* vec1, float* vec2, float* ansVec, int vectorLen);

/**
 * @brief sums the even indexes and pows the odd indexes of the second vector
 * @param vec1 first vector to add
 * @param vec2 second vector to add or pow
 * @param ansVec the vector that saves the answers
 * @param vectorLen the vectors length
 */
__global__ void vecAddEvenMulOdd(float* vec1, float* vec2, float* ansVec, int vectorLen);

/**
 * @brief each thread does addition of even indexes and multiplication of odd indexes of the second vector
 * @param vec1 first vector to add or multipy
 * @param vec2 second vector to add or multipy
 * @param ansVec the vector that saves the answers
 * @param vectorLen the vectors length
 */
__global__ void vec2Operations(float* vec1, float* vec2, float* ansVec, int vectorLen);

/**
 * @brief each thread does n addittions
 * @param vec1 first vector to add 
 * @param vec2 second vector to add 
 * @param ansVec the vector that saves the answers
 * @param vectorLen the vectors length
 * @param n the number of operations each thread performs
 */
__global__ void vecNOperations(float* vec1, float* vec2, float* ansVec, int vectorLen, int n);

/**
 * @brief each thread does n addittionswhere every result uses the corresponding element from vec1 and an element from vec2 accessed with stride n (interleaved layout).
 * @param vec1 first vector to add 
 * @param vec2 second vector to add 
 * @param ansVec the vector that saves the answers
 * @param vectorLen the vectors length
 * @param n the number of operations each thread performs
 */
__global__ void vecNOpFlip(float* vec1, float* vec2, float* ansVec, int vectorLen, int n);