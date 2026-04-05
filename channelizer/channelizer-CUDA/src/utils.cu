#include "../Include/utils.cuh"

//initialize vector with values from file
void initVecFile(std::vector<cuComplex>& vec, const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filePath + "'");
    }
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    int numElements = fileSize / sizeof(cuComplex);
    vec.resize(numElements);
    if (!file.read(reinterpret_cast<char*>(vec.data()), fileSize)) {
        throw std::runtime_error("Error: Could not read all data from '" + filePath + "'");
    }

    file.close();
    std::cout << "Successfully loaded " << numElements << " complex samples from " << filePath << std::endl;
}

//initialize vector with values from file
void initRealVecFile(std::vector<float>& vec, const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filePath + "'");
    }
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    int numElements = fileSize / sizeof(float);
    vec.resize(numElements);
    if (!file.read(reinterpret_cast<char*>(vec.data()), fileSize)) {
        throw std::runtime_error("Error: Could not read all data from '" + filePath + "'");
    }

    file.close();
    std::cout << "Successfully loaded " << numElements << " complex samples from " << filePath << std::endl;
}

//write the content of a vector to a file
void ansToFile(const std::string& filePath, cuComplex* vec, int length)
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error(std::string
        ("Error: Could not create file '") + filePath + "'");
    }
    file.write(reinterpret_cast<const char*>(vec), length * sizeof(cuComplex));
    file.close();

    std::cout << "File '" << filePath << "' created with " << length << " float values." << std::endl;
}

void polyphaseParams(float fsHz, float channelBw, float overlapFactor, int filterLen, int inputLen,
     int& numBlocks, int& numChannels, int& decimationFactor, int& olaParam)
{
    int requested_channels_num = static_cast<int>(fsHz / channelBw);
    decimationFactor = static_cast<int>(requested_channels_num / overlapFactor);

    numChannels = int(requested_channels_num * overlapFactor);
       
    if (filterLen % numChannels != 0){
        throw std::runtime_error(std::string
        ("Error: filter size is not divisible by num_channels, Check filter design parameters"));
    }
    olaParam = filterLen / numChannels;

    if (inputLen < filterLen) {
        numBlocks = 0;
    } else {
        numBlocks = (inputLen - filterLen) / decimationFactor + 1;
    }
    std::cout << "[PARAMS] M=" << numChannels << ", D=" << decimationFactor 
              << ", P=" << olaParam << ", Blocks=" << numBlocks << std::endl;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}