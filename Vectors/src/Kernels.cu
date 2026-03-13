#include "../Include/Kernels.cuh"

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
        ansVec[workIndex] = (workIndex % 2 == 0) ? (vec1[workIndex] + vec2[workIndex]) : (vec2[workIndex] * vec2[workIndex]);
    }
}

__global__ void vec2Operations(float* vec1, float* vec2, float* ansVec, int vectorLen)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = vectorLen / 2;
    if(workIndex < stride)
    {
        ansVec[workIndex] = vec1[workIndex] + vec2[workIndex];
        int secondIndex = workIndex + stride;
        ansVec[secondIndex] = vec1[secondIndex] * vec2[secondIndex];
    }
}

__global__ void vecNOperations(float* vec1, float* vec2, float* ansVec, int vectorLen, int n)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = vectorLen / n;
    for(int i = 0; i < n; i++)
    {
        int currentIndex = workIndex + i * stride;
        if(currentIndex < vectorLen)
        {
             ansVec[currentIndex] = vec1[currentIndex] + vec2[currentIndex];   
        }
    }
}

__global__ void vecNOpFlip(float* vec1, float* vec2, float* ansVec, int vectorLen, int n)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = vectorLen / n;
    for(int i = 0; i < n; i++)
    {
       int currentIndex = workIndex + i * stride;
        if (currentIndex < vectorLen)
        {
            int blockStart = currentIndex - (currentIndex % n);
            int offsetInBlock = currentIndex % n;
            int vec2Index = blockStart + (n - 1 - offsetInBlock);  
            ansVec[currentIndex] = vec1[currentIndex] + vec2[vec2Index];
        }
    }
}