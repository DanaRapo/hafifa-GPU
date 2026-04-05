#include "../Include/PolyphaseDirect.cuh"
#include "../Include/PolyphaseCublas.cuh"
#include <iostream>

int main() {
    PolyphaseConfig no_overlap;
    no_overlap.inputPath  = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/input_signal.bin";
    no_overlap.filterPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer.bin";
    no_overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/direct_no_overlap_res_wola.bin";
    no_overlap.fsHz = 10000.0f;
    no_overlap.bwHz = 1000.0f;
    no_overlap.overlap = 1.0f;

    PolyphaseConfig overlap;
    overlap.inputPath  = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/input_signal_overlap_dense.bin";
    overlap.filterPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer_overlap.bin";
    overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/direct_overlap_res_wola.bin";
    overlap.fsHz = 10000.0f;
    overlap.bwHz = 1000.0f;
    overlap.overlap = 2.0f;

    PolyphaseConfig cublas_no_overlap;
    cublas_no_overlap.inputPath  = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/input_signal.bin";
    cublas_no_overlap.filterPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer.bin";
    cublas_no_overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/cublas_no_overlap_res_wola.bin";
    cublas_no_overlap.fsHz = 10000.0f;
    cublas_no_overlap.bwHz = 1000.0f;
    cublas_no_overlap.overlap = 1.0f;

    PolyphaseConfig cublas_overlap;
    cublas_overlap.inputPath  = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/input_signal_overlap_dense.bin";
    cublas_overlap.filterPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/filter_from_designer_overlap.bin";
    cublas_overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/cublas_overlap_res_wola.bin";
    cublas_overlap.fsHz = 10000.0f;
    cublas_overlap.bwHz = 1000.0f;
    cublas_overlap.overlap = 2.0f;

    try {
        std::cout << "--- Running Scenario: No Overlap ---" << std::endl;
        runCompletePolyphasePipeline(no_overlap);

        std::cout << "\n--- Running Scenario: Overlap ---" << std::endl;
        runCompletePolyphasePipeline(overlap);

        std::cout << "\n--- Running Scenario: cuBLAS No Overlap ---" << std::endl;
        runCompleteCublasPipeline(cublas_no_overlap);
        
        std::cout << "\n--- Running Scenario: cuBLAS Overlap ---" << std::endl;
        runCompleteCublasPipeline(cublas_overlap);
    } 
    catch (const std::exception& e) {
        std::cerr << "Execution failed: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}