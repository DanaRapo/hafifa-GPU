#include "../Include/utils.h"
//intialize vector with values from 0 to length-1 multiplied by factor
void initVec(float* vec, int length, int factor)
{
    for(int i = 0; i<length; i++)
    {
        vec[i] = i * factor;
    }
}

//initialize vector with values from file
void initVecFile(float* vec, int length,const std::string& filePath)
{
    std::ifstream file(filePath,  std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string
        ("Error: Could not open file '") + filePath + "'");
    }
    file.read(reinterpret_cast<char*>(vec), length * sizeof(float));
    file.close();
}

//create the contect of a file 32f
void create32File(const std::string& filePath , int length, int factor)
{
    std::vector<float> data(length);
    for (int i = 0; i < length; i++) {
        data[i] = static_cast<float>(i * factor) ;
    }
    std::ofstream file(filePath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error(std::string
        ("Error: Could not create file '") + filePath + "'");
    }
    file.write(reinterpret_cast<const char*>(data.data()), length * sizeof(float));
    file.close();

    std::cout << "File '" << filePath << "' created with " << length << " float values." << std::endl;
}

//write the content of a vector to a file
void ansToFile(const std::string& filePath, float* vec, int length)
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error(std::string
        ("Error: Could not create file '") + filePath + "'");
    }
    file.write(reinterpret_cast<const char*>(vec), length * sizeof(float));
    file.close();

    std::cout << "File '" << filePath << "' created with " << length << " float values." << std::endl;
}

//read the content of a file and print it to the console
void printFileContent(const std::string& filePath, int length)
{
    std::vector<float> data(length);
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string
        ("Error: Could not open file '") + filePath + "'");
    }
    file.read(reinterpret_cast<char*>(data.data()), length * sizeof(float));
    file.close();
    std::cout << "Content of file '" << filePath << "':\n";
    for (int i = 0; i < length; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}
