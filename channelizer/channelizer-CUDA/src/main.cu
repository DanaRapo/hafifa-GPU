#include "../Include/ChannelizerDirect.cuh"
#include "../Include/ChannelizerCublas.cuh"
#include <iostream>
#include <memory>

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

    PolyphaseConfig cublas_no_overlap = no_overlap;
    cublas_no_overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/cublas_no_overlap_res_wola.bin";

    PolyphaseConfig cublas_overlap = overlap;
    cublas_overlap.outputPath = "/home/test3/Desktop/git/hafifa-GPU/channelizer/channelizer-python/signals/cublas_overlap_res_wola.bin";
    try {
        // --- Direct Scenarios ---
        
        {
            std::cout << "--- Running Scenario: Direct No Overlap ---" << std::endl;
            auto direct_no = ChannelizerDirect::createAndLoad(no_overlap);
            if (direct_no) direct_no->runCompleteDirectPipeline(no_overlap);
        } // 'direct_no' is destroyed here, GPU memory freed

        {
            std::cout << "\n--- Running Scenario: Direct Overlap ---" << std::endl;
            auto direct_ov = ChannelizerDirect::createAndLoad(overlap);
            if (direct_ov) direct_ov->runCompleteDirectPipeline(overlap);
        } // 'direct_ov' is destroyed here, GPU memory freed

        // --- cuBLAS Scenarios ---

        {
            std::cout << "\n--- Running Scenario: cuBLAS No Overlap ---" << std::endl;
            auto cublas_no = ChannelizerCublas::createAndLoad(cublas_no_overlap);
            if (cublas_no) cublas_no->runCompleteCublasPipeline(cublas_no_overlap);
        } // 'cublas_no' is destroyed here, GPU memory freed

        {
            std::cout << "\n--- Running Scenario: cuBLAS Overlap ---" << std::endl;
            auto cublas_ov = ChannelizerCublas::createAndLoad(cublas_overlap);
            if (cublas_ov) cublas_ov->runCompleteCublasPipeline(cublas_overlap);
        } // 'cublas_ov' is destroyed here, GPU memory freed

    } 
    catch (const std::exception& e) {
        std::cerr << "Critical Error during execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}