#include "../Include/PolyphaseDirect.cuh"
#include "../Include/PolyphaseDirect.cuh"
#include "../Include/KernelPolyphaseDirect.cuh"
#include "../Include/utils.cuh"
#include <iostream>
#include <vector>

void runCompletePolyphasePipeline(PolyphaseConfig& cfg) {
    std::vector<cuComplex> hData;
    std::vector<float> hFilterReal;
    
    try{
        initVecFile(hData, cfg.inputPath);
        initRealVecFile(hFilterReal, cfg.filterPath);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing data: " << e.what() << std::endl;
        return;
    }

    std::vector<cuComplex> h_filter_complex(hFilterReal.size());
    for (size_t i = 0; i < hFilterReal.size(); ++i) {
        h_filter_complex[i] = make_cuComplex(hFilterReal[i], 0.0f);
    }

    polyphaseParams(cfg.fsHz, cfg.bwHz, cfg.overlap, (int)h_filter_complex.size(), (int)hData.size(),
                     cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

    cuComplex *d_input, *d_filter, *d_output;
    checkCuda(cudaMallocManaged(&d_input, hData.size() * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&d_filter, h_filter_complex.size() * sizeof(cuComplex)));
    checkCuda(cudaMallocManaged(&d_output, cfg.numChannels * cfg.numBlocks * sizeof(cuComplex)));

    memcpy(d_input, hData.data(), hData.size() * sizeof(cuComplex));
    memcpy(d_filter, h_filter_complex.data(), h_filter_complex.size() * sizeof(cuComplex));

    runWolaDirect(d_input, d_filter, d_output, 
                   cfg.numBlocks, cfg.numChannels, cfg.decimation, cfg.olaParam);

    checkCuda(cudaDeviceSynchronize());
    try{
        ansToFile(cfg.outputPath, d_output, cfg.numBlocks * cfg.numChannels);
    } catch (const std::exception& e) {
        std::cerr << "Error saving output: " << e.what() << std::endl;
    }
    std::cout << "Successfully saved: " << cfg.outputPath << std::endl;

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}

void runWolaDirect(cuComplex* input, cuComplex* filter, cuComplex* output,
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
