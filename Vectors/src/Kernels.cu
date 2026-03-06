#include "Kernels.cuh"

__global__ void vecAdd(float* vec1, float* vec2, float* ansVec, int vectorLen)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLen)
    {
        ansVec[workIndex] = vec1[workIndex] + vec2[workIndex];
    }
} 

__global__ void vecAddEvenMulOdd(float* vec1, float* vec2, float* ansVec, int vectorLen)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLen)
    {
        if(workIndex % 2 == 0)
        {
            ansVec[workIndex] = vec1[workIndex] + vec2[workIndex];
        }
        else
        {
            ansVec[workIndex] = vec2[workIndex] * vec2[workIndex];
        }
    }
}

__global__ void vec2Operations(float* vec1, float* vec2, float* ansVec, int vectorLen)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int evenIndex = workIndex * 2;
    int oddIndex = workIndex * 2 + 1;
    if(evenIndex < vectorLen)
    {
        ansVec[evenIndex] = vec1[evenIndex] + vec2[evenIndex];
    }
    if(oddIndex < vectorLen)
    {
        ansVec[oddIndex] = vec1[oddIndex] * vec2[oddIndex];
    }
}

__global__ void vecNOperations(float* vec1, float* vec2, float* ansVec, int vectorLen, int n)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int startIndex = workIndex * n;
    for(int i = 0; i < n; i++)
    {
        int currentIndex = startIndex + i;
        if(currentIndex >= vectorLen)
        {
            return;
        }
        ansVec[currentIndex] = vec1[currentIndex] + vec2[currentIndex];
    }
}

__global__ void vecNOpFlip(float* vec1, float* vec2, float* ansVec, int vectorLen, int n)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int startIndex = workIndex * n;
    for(int i = 0; i < n; i++)
    {
        int currentIndex = startIndex + i;
        int vec2Index = startIndex + (n - 1 - i);
        if(currentIndex >= vectorLen || vec2Index >= vectorLen)
        {
            return;
        }
        ansVec[currentIndex] = vec1[currentIndex] + vec2[vec2Index];
    }
}