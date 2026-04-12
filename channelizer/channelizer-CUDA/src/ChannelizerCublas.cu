#include "../Include/ChannelizerCublas.cuh"

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
ChannelizerCublas::ChannelizerCublas() : _handle(nullptr), _dInput(nullptr), _dFilter(nullptr), _dOutput(nullptr), _dDataMat(nullptr), _dPhaseRes(nullptr) {
    cublasCreate(&_handle);
}
std::unique_ptr<ChannelizerCublas> ChannelizerCublas::createAndLoad(PolyphaseConfig& cfg) {
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

        auto instance = std::unique_ptr<ChannelizerCublas>(
            new ChannelizerCublas(cfg, hData.size(), hFilter.size())
        );

        instance->m_hData = std::move(hData);
        instance->m_hFilter = std::move(hFilter);

        return instance;

    } catch (const std::exception& e) {
        std::cerr << "[Channelizer Error] Failed to create instance: " << e.what() << std::endl;
        return nullptr;
    }
}

ChannelizerCublas::ChannelizerCublas(const PolyphaseConfig& cfg, size_t inputSize, size_t filterSize) 
    : _handle(nullptr), _dInput(nullptr), _dFilter(nullptr), _dOutput(nullptr), _dDataMat(nullptr), _dPhaseRes(nullptr) 
{
    cublasCreate(&_handle);

    m_outSize = (size_t)cfg.numChannels * cfg.numBlocks;
    size_t matSize = m_outSize * cfg.olaParam;

    checkCuda(cudaMallocManaged(&_dInput, inputSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dFilter, filterSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dOutput, m_outSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dDataMat, matSize * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&_dPhaseRes, m_outSize * sizeof(cuComplex)));
}

ChannelizerCublas::~ChannelizerCublas() {
    if (_dInput) cudaFree(_dInput);
    if (_dFilter) cudaFree(_dFilter);
    if (_dOutput) cudaFree(_dOutput);
    if (_dDataMat) cudaFree(_dDataMat);
    if (_dPhaseRes) cudaFree(_dPhaseRes);
    if (_handle) cublasDestroy(_handle);
}

void ChannelizerCublas::runWola(cublasHandle_t handle, cuComplex* inputStream, cuComplex* filterMat, 
             cuComplex* output, cuComplex* dataMat, cuComplex* phaseRes,
             int numBlocks, int numChannels, int decimationFactor, int olaParam)
{
    size_t totalElementsPerPhase = (size_t)numChannels * numBlocks;

    dim3 block(16, 16);
    dim3 grid((numChannels + block.x - 1) / block.x, (numBlocks + block.y - 1) / block.y);

    PrepareDataCublas<<<grid, block>>>(inputStream, dataMat, numChannels, decimationFactor, numBlocks, olaParam);

    checkCuda(cudaMemset(output, 0, totalElementsPerPhase * sizeof(cuComplex)));
    
    cuComplex alpha = {1.0f, 0.0f};
   
    for (int i = 0; i < olaParam; i++) {
        cuComplex* currentData = dataMat + (i * totalElementsPerPhase);
        cuComplex* currentFilter = filterMat + (i * numChannels);

        cublasCdgmm(handle, 
                    CUBLAS_SIDE_LEFT, 
                    numChannels, numBlocks, 
                    currentData, numChannels, 
                    currentFilter, 1, 
                    phaseRes, numChannels);

        cublasCaxpy(handle, (int)totalElementsPerPhase, &alpha, phaseRes, 1, output, 1);
    }
}

void ChannelizerCublas::runCompleteCublasPipeline(const PolyphaseConfig& cfg) {
    // Corrected variable names with underscores
    checkCuda(cudaMemcpy(_dInput, m_hData.data(), m_hData.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(_dFilter, m_hFilter.data(), m_hFilter.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Warm-up run
    runWola(_handle, _dInput, _dFilter, _dOutput, _dDataMat, _dPhaseRes,
            cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

    checkCuda(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        runWola(_handle, _dInput, _dFilter, _dOutput, _dDataMat, _dPhaseRes, 
                cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\n[BENCHMARK] Avg Execution Time: %f ms\n", milliseconds / iterations);

    try {
        ansToFile(cfg.outputPath, _dOutput, cfg.numBlocks * cfg.numChannels);
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
