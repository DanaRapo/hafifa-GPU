#include "../Include/KernelPolyphaseDirect.cuh"

__global__ void wolaDirect( const cuComplex* __restrict__ input, const cuComplex* __restrict__ filter,
    cuComplex* output, int olaParam, int numChannels, int decimationFactor, int numBlocks)
{
    int chanIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int TimeblockIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( chanIdx >= numChannels || TimeblockIdx >= numBlocks )
       return;

    extern __shared__ cuComplex s_filter[];
    
    int localChan = threadIdx.x;
    if(threadIdx.y == 0) {
       for (int load = 0; load < olaParam; load++) {
            int globalFilterIdx = load * numChannels + chanIdx;
            int localFilterIdx = localChan * olaParam + load;

            if (chanIdx < numChannels) {
                s_filter[localFilterIdx] = filter[globalFilterIdx];
            }
        }
    }
    
    __syncthreads();

    cuComplex accumulator = make_cuComplex(0.0f, 0.0f);

    int chanFlip = (numChannels - 1) - chanIdx;
    cuComplex* s_filter_ptr = &s_filter[localChan * olaParam];
    #pragma unroll
    for( int i = 0; i < olaParam; i++ )
    {
        int i_flip = (olaParam - 1) - i;
        int inputIdx = (TimeblockIdx * decimationFactor) + i_flip * numChannels + chanFlip;
        cuComplex sample = input[inputIdx];

        cuComplex filterCoeff = s_filter_ptr[i];
        accumulator = cuCaddf(accumulator, cuCmulf(sample, filterCoeff));
    }
    int outputIdx = TimeblockIdx * numChannels + chanIdx;
    output[outputIdx] = accumulator;
}