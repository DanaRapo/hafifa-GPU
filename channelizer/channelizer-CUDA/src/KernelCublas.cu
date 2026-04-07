#include "../Include/KernelCublas.cuh"

__global__ void PrepareDataCublas(cuComplex* inputStream, cuComplex* dataMat,
     int numChannels, int decimationFactor, int numBlocks, int olaParam)
{
    int chanIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int TimeBlockIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (chanIdx < numChannels && TimeBlockIdx < numBlocks) {
        int channelFlip = (numChannels - 1 - chanIdx);
        #pragma unroll
        for( int i = 0; i < olaParam; i++)
        {
            int flipped_i = (olaParam - 1 - i);
            
            int inputIdx = TimeBlockIdx * decimationFactor + flipped_i * numChannels + channelFlip;
            int outIdx = i * (numChannels * numBlocks) + (TimeBlockIdx * numChannels + chanIdx);
            dataMat[outIdx] = inputStream[inputIdx];
        }
    }
}