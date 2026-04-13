#include "../Include/ChannelizerDirect.cuh"

__global__ void wolaDirect( const cuComplex* __restrict__ input, const cuComplex* __restrict__ filter,
    cuComplex* output, int olaParam, int numChannels, int decimationFactor, int numBlocks)
{
    int chanIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int TimeblockIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if( chanIdx >= numChannels || TimeblockIdx >= numBlocks )
       return;

    extern __shared__ cuComplex s_filter[];
    
    int localChan = threadIdx.x;
    
    for (int load = 0; load < olaParam; load++) {
        int globalFilterIdx = load * numChannels + chanIdx;
        int localFilterIdx = localChan * olaParam + load;

        if (chanIdx < numChannels) {
            s_filter[localFilterIdx] = filter[globalFilterIdx];
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

ChannelizerDirect::ChannelizerDirect() : _dInput(nullptr), _dFilter(nullptr), _dOutput(nullptr) {}

ChannelizerDirect::~ChannelizerDirect() {
    if (_dInput) cudaFree(_dInput);
    if (_dFilter) cudaFree(_dFilter);
    if (_dOutput) cudaFree(_dOutput);
}

std::unique_ptr<ChannelizerDirect> ChannelizerDirect::createAndLoad(PolyphaseConfig& cfg) {
    std::vector<cuComplex> hData;
    std::vector<float> hFilterReal;

    try {
        initVecFile(hData, cfg.inputPath);
        initRealVecFile(hFilterReal, cfg.filterPath);

        std::vector<cuComplex> hFilter(hFilterReal.size());
        for (size_t i = 0; i < hFilterReal.size(); ++i) {
            hFilter[i] = make_cuComplex(hFilterReal[i], 0.0f);
        }

        polyphaseParams(cfg.fsHz, cfg.bwHz, cfg.overlap, (int)hFilter.size(), (int)hData.size(),
                         cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

        auto instance = std::unique_ptr<ChannelizerDirect>(
            new ChannelizerDirect(cfg, hData.size(), hFilter.size())
        );

        instance->m_hData = std::move(hData);
        instance->m_hFilter = std::move(hFilter);

        return instance;
    } catch (const std::exception& e) {
        std::cerr << "[Direct Error] " << e.what() << std::endl;
        return nullptr;
    }
}

ChannelizerDirect::ChannelizerDirect(const PolyphaseConfig& cfg, size_t inputSize, size_t filterSize) 
    : _dInput(nullptr), _dFilter(nullptr), _dOutput(nullptr) 
{
    m_outSize = (size_t)cfg.numChannels * cfg.numBlocks;
    
    // Shared memory size calculation (channels in block * olaParam)
    // Assuming 16 threads in X dimension as per threadsPerBlock
    m_sharedMemSize = (16 * cfg.olaParam) * sizeof(cuComplex);

    checkCuda(cudaMallocManaged(&_dInput, inputSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dFilter, filterSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dOutput, m_outSize * sizeof(cuComplex)));
}

void ChannelizerDirect::runWolaDirect(cuComplex* input, cuComplex* filter, cuComplex* output,
        int numBlocks, int numChannels, int decimationFactor, int olaParam) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocksInKernel(
        (numChannels + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (numBlocks + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    int shared_mem_size = (numChannels * olaParam) * sizeof(cuComplex);
    
    wolaDirect<<<numBlocksInKernel, threadsPerBlock, shared_mem_size>>>(
        input, filter, output, olaParam, numChannels, decimationFactor, numBlocks);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in WOLA Kernel: " << cudaGetErrorString(err) << std::endl;
    }
}

void ChannelizerDirect::runCompleteDirectPipeline(PolyphaseConfig& cfg) {
    checkCuda(cudaMemcpy(_dInput, m_hData.data(), m_hData.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(_dFilter, m_hFilter.data(), m_hFilter.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Warm-up
    runWolaDirect(_dInput, _dFilter, _dOutput, cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);
    checkCuda(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        runWolaDirect(_dInput, _dFilter, _dOutput, cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\n[BENCHMARK] Direct M=%d, Avg Time: %f ms\n", cfg.numChannels, ms / iterations);

    try {
        ansToFile(cfg.outputPath, _dOutput, m_outSize);
    } catch (const std::exception& e) {
        std::cerr << "Save Error: " << e.what() << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

