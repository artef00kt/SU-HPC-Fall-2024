

#include <vector>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h> 
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

static const int blockSize = 1024;
static const int gridSize = 12;

cudaError_t vectorSumWithCUDA(int* res, const int* vector, unsigned int size);
int vectorSumWithCPU(int* res, int* vector, unsigned int size);

__global__ void vectorSumKernel(const int* inArr, int* outArr, unsigned int arraySize)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockSize; // глобальный индекс потока
    const int gridSize = blockSize * gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += inArr[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        outArr[blockIdx.x] = shArr[0];
}

// функция для генерации псевдослучайного массива
int* createRandArray(int size) {
    const int max = 0;
    const int min = 10;
    int* array = new int[size];
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % (min - max + 1) + max;
    }

    return array;
}

// функция для выводы массива в консоль в виде строки
void printArray(int* a, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl;
}


int main()
{
    unsigned int size;
    std::cout << "Input vector size N" << std::endl << "N: ";
    std::cin >> size;

    // выделение всей нужной памяти и генерация вектора
    int* vector = createRandArray(size);
    int* resultCPU = new int;
    int* resultGPU = new int;

    // вычисление произведения матриц на CPU с замером времени
    clock_t startCPU = clock();
    vectorSumWithCPU(resultCPU, vector, size);
    clock_t endCPU = clock();
    double timeCPU = endCPU - startCPU;

    std::cout << "Time on CPU: " << timeCPU / CLOCKS_PER_SEC * 1000.0 << " msec." << std::endl;
    
    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorSumWithCUDA(resultGPU, vector, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    std::cout << "Result CPU:" << *resultCPU << std::endl;
    std::cout << "Result GPU:" << *resultGPU << std::endl;

    delete[] vector;
    delete resultCPU;
    delete resultGPU;

    return 0;
}

int vectorSumWithCPU(int* res, int* vector, unsigned int size) {
    *res = 0;
    for (int i = 0; i < size; ++i) {
        *res += vector[i];
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t vectorSumWithCUDA(int* res, const int* vector, unsigned int size)
{
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    cudaError_t cudaStatus;

    cudaEvent_t startTime, endTime;
    float time;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, gridSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, vector, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaEventRecord(startTime, 0);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(16, 16); // 16 * 16 = 256 threads in block
    dim3 numBlocks(ceil(double(size) / threadsPerBlock.x), ceil(double(size) / threadsPerBlock.y));
    // vectorSumKernel<<<numBlocks, threadsPerBlock>>> (dev_a, dev_b, size);
    vectorSumKernel <<<gridSize, blockSize>>> (dev_a, dev_b, size);
    vectorSumKernel <<<1, blockSize>>> (dev_b, dev_b, gridSize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventRecord(endTime, 0);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(res, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventElapsedTime(&time, startTime, endTime);

    std::cout << "Time on GPU: " << time << " msec." << std::endl;

Error:
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}