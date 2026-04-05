#include "../Include/PolyphaseCublas.cuh"

void runWola(cublasHandle_t handle, cuComplex* inputStream, cuComplex* filterMat, 
              cuComplex* output, int numBlocks, int numChannels, int decimationFactor, int olaParam)
{
    cuComplex* dataMat;
    size_t totalElementsPerPhase = (size_t)numChannels * numBlocks;
    size_t totalMatSize = totalElementsPerPhase * olaParam;
    
    checkCuda(cudaMallocManaged(&dataMat, totalMatSize * sizeof(cuComplex)));
   
    cuComplex* phaseRes;
    checkCuda(cudaMallocManaged(&phaseRes, totalElementsPerPhase * sizeof(cuComplex)));

    dim3 block(16, 16);
    dim3 grid((numChannels + block.x - 1) / block.x, (numBlocks + block.y - 1) / block.y);

    PrepareDataCublas<<<grid, block>>>(inputStream, dataMat, numChannels, decimationFactor, numBlocks, olaParam);

    checkCuda(cudaMemset(output, 0, totalElementsPerPhase * sizeof(cuComplex)));
    checkCuda(cudaDeviceSynchronize());

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

    checkCuda(cudaFree(dataMat));
    checkCuda(cudaFree(phaseRes));
}

void runCompleteCublasPipeline(PolyphaseConfig& cfg) {
    std::vector<cuComplex> hData;
    std::vector<float> hFilterReal;
    
    try{
        initVecFile(hData, cfg.inputPath);
        initRealVecFile(hFilterReal, cfg.filterPath);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing data: " << e.what() << std::endl;
        return;
    }
 
    std::vector<cuComplex> hFilterComplex(hFilterReal.size());
    for (size_t i = 0; i < hFilterReal.size(); ++i) {
        hFilterComplex[i] = make_cuComplex(hFilterReal[i], 0.0f);
    }

    polyphaseParams(cfg.fsHz, cfg.bwHz, cfg.overlap, (int)hFilterComplex.size(), (int)hData.size(),
                     cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cuComplex *d_input, *d_filter, *d_output;
    checkCuda(cudaMallocManaged(&d_input, hData.size() * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&d_filter, hFilterComplex.size() * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&d_output, cfg.numChannels * cfg.numBlocks * sizeof(cuComplex)));

    checkCuda(cudaMemcpy(d_input, hData.data(), hData.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_filter, hFilterComplex.data(), hFilterComplex.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));

    runWola(handle, d_input, d_filter, d_output, 
             cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

    checkCuda(cudaDeviceSynchronize());
    try{
        ansToFile(cfg.outputPath, d_output, cfg.numBlocks * cfg.numChannels);
    } catch (const std::exception& e) {
        std::cerr << "Error saving output: " << e.what() << std::endl;
    }
    std::cout << "cuBLAS Result saved: " << cfg.outputPath << std::endl;

    cublasDestroy(handle);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}