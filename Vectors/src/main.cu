#include <cuda_runtime.h>
#include "../Include/Kernels.cuh"
#include "../Include/utils.h"

//check for cuda errors
void checkCuda(cudaError_t result) 
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int main()
{
    int length = 1024;
    int operationNum = 4;
    int size = length * sizeof(float);
    try{
        create32File("vector_data.32f", length, 2);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    //memory allocation for host and device vectors
    float *host_vec1, *host_vec2, *host_ansAdd,
     *host_ansEvenMulOdd, *host_ans2Op, *host_ansNop , *host_ansNopFlip;

    host_vec1 = new float[length];
    host_vec2 = new float[length];
    host_ansAdd = new float[length];
    host_ansEvenMulOdd = new float[length];
    host_ans2Op = new float[length];
    host_ansNop = new float[length];
    host_ansNopFlip = new float[length];

    //initialize host vectors
   initVec(host_vec1,length,1);
   try{
        initVecFile(host_vec2,length,"vector_data.32f");
    }
    catch (const std::exception& e) {
          std::cerr << "Error: " << e.what() << std::endl;
          return 1;
    }
   
   //initVec(host_vec2,length,2);

    float *dev_vec1, *dev_vec2, *dev_ansAdd,
     *dev_ansEvenMulOdd, *dev_ans2Op, *dev_ansNop, *dev_ansNopFlip;

    checkCuda(cudaMalloc((void**)&dev_vec1, size));
    checkCuda(cudaMalloc((void**)&dev_vec2, size));
    checkCuda(cudaMalloc((void**)&dev_ansAdd, size));
    checkCuda(cudaMalloc((void**)&dev_ansEvenMulOdd, size));
    checkCuda(cudaMalloc((void**)&dev_ans2Op, size));
    checkCuda(cudaMalloc((void**)&dev_ansNop, size));
    checkCuda(cudaMalloc((void**)&dev_ansNopFlip, size));

    //copy host vectors to device
    checkCuda(cudaMemcpy(dev_vec1, host_vec1, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_vec2, host_vec2, size, cudaMemcpyHostToDevice));

    //define the number of threads to the first kernels and calculate the number of blocks needed to cover all elements
    int threadsPerBlock = 256;
    int blockPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    //define the number of threads to the third kernel- each thread does 2 operations
    int numThreads2Op = length / 2;
    int numBlocks2Op = (numThreads2Op + threadsPerBlock - 1) / threadsPerBlock;
    //define the number of threads to the fourth kernel- each thread does n operations
    int numThreadsNop = length / operationNum;
    int numBlocksNop = (numThreadsNop + threadsPerBlock - 1) / threadsPerBlock;

    vecAdd<<<blockPerGrid, threadsPerBlock>>>(dev_vec1, dev_vec2, dev_ansAdd, length);
    vecAddEvenMulOdd<<<blockPerGrid, threadsPerBlock>>>(dev_vec1, dev_vec2, dev_ansEvenMulOdd, length);
    vec2Operations<<<numBlocks2Op, threadsPerBlock>>>(dev_vec1, dev_vec2, dev_ans2Op, length);
    vecNOperations<<<numBlocksNop, threadsPerBlock>>>(dev_vec1, dev_vec2, dev_ansNop, length, operationNum);
    vecNOpFlip<<<numBlocksNop, threadsPerBlock>>>(dev_vec1, dev_vec2, dev_ansNopFlip, length, operationNum);
    cudaDeviceSynchronize();
    //copy results back to host
    checkCuda(cudaMemcpy(host_ansAdd, dev_ansAdd, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host_ansEvenMulOdd, dev_ansEvenMulOdd, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host_ans2Op, dev_ans2Op, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host_ansNop, dev_ansNop, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(host_ansNopFlip, dev_ansNopFlip, size, cudaMemcpyDeviceToHost));

    try{
        ansToFile("vector_result.32f", host_ansAdd, length);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    try{
        printFileContent("vector_result.32f", 10);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    //print the first 10 results of each operation
    std::cout<<"Result example:\n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << host_vec1[i] << " + " << host_vec2[i] << " = " << host_ansAdd[i] << std::endl;
        std::cout << host_vec1[i] << " + " << host_vec2[i] << " = " << host_ansNop[i] << std::endl;
        std::cout << " flipped answer "<< i  << ": " << host_ansNopFlip[i] << std::endl;
        if(i%2==0)
        {
            std::cout << host_vec1[i] << " + " << host_vec2[i] << " = " << host_ansEvenMulOdd[i] << std::endl;
            std::cout << host_vec1[i] << " + " << host_vec2[i] << " = " << host_ans2Op[i] << std::endl;
        }
        else
        {
             std::cout << host_vec2[i] << " ^ 2  = " << host_ansEvenMulOdd[i] << std::endl;
             std::cout << host_vec1[i] << " * " << host_vec2[i] << " = " << host_ans2Op[i] << std::endl;
        }
    }
    //free device memory
    checkCuda(cudaFree(dev_vec1));
    checkCuda(cudaFree(dev_vec2));
    checkCuda(cudaFree(dev_ansAdd));
    checkCuda(cudaFree(dev_ansEvenMulOdd));
    checkCuda(cudaFree(dev_ans2Op));
    checkCuda(cudaFree(dev_ansNop));
    checkCuda(cudaFree(dev_ansNopFlip));
    //free host memory
    delete[] host_vec1;
    delete[] host_vec2;
    delete[] host_ansAdd;
    delete[] host_ansEvenMulOdd;
    delete[] host_ans2Op;
    delete[] host_ansNop;
    delete[] host_ansNopFlip;

    return 0;
}