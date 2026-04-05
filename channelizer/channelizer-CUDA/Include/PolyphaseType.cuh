#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

struct PolyphaseConfig {
    std::string inputPath;
    std::string filterPath;
    std::string outputPath;
    float fsHz;
    float bwHz;
    float overlap;

    int numBlocks;
    int numChannels;
    int decimation;
    int olaParam;
};