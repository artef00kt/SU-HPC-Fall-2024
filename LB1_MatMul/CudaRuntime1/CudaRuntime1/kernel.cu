#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h> 

cudaError_t multiplyMatrixWithCUDA(int *res, const int * matrix1, const int * matrix2, unsigned int size);
int multiplyMatrixWithCPU(int* res, int* matrix1, int* matrix2, unsigned int size);

__global__ void multiplyMatrixKernel(int *c, const int *a, const int *b, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        c[i * size + j] = 0;
        for (int k = 0; k < size; ++k) {
            c[i * size + j] += a[i * size + k] * b[k * size + j];
        }
    }
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

// функция для сравнения двух массивов
bool arraysEqual(int* a, int* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// функция для выводы массива в консоль в виде строки (нужно было для дебага)
void printArray(int* a, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl;
}

// функция для выводы массива в консоль в виде матрицы (нужно было для дебага)
void printArrayLikeMatrix(int* a, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << a[i * size + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    unsigned int size;
    std::cout << "Input matrix size N x N" << std::endl << "N: ";
    std::cin >> size;

    // выделение всей нужной памяти и генерация матриц
    int* matrix1 = createRandArray(size * size);
    int* matrix2 = createRandArray(size * size);
    int* resultCPU = new int[size * size];
    int* resultGPU = new int[size * size];

    // вычисление произведения матриц на CPU с замером времени
    clock_t startCPU = clock();
    multiplyMatrixWithCPU(resultCPU, matrix1, matrix2, size);
    clock_t endCPU = clock();
    double timeCPU = endCPU - startCPU;

    std::cout << "Time on CPU: " << timeCPU / CLOCKS_PER_SEC * 1000.0 << " msec." << std::endl;

    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyMatrixWithCUDA(resultGPU, matrix1, matrix2, size);
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

    // printArray(matrix1, size * size);
    // printArrayLikeMatrix(matrix1, size);
    // printArrayLikeMatrix(matrix2, size);
    // printArray(resultCPU, size * size);
    // printArray(resultGPU, size * size);
    // printArrayLikeMatrix(resultCPU, size);
    // printArrayLikeMatrix(resultGPU, size);

    std::cout << "Matrix is equal: " << (arraysEqual(resultCPU, resultGPU, size * size) ? "true" : "false");

    delete[] matrix1;
    delete[] matrix2;
    delete[] resultCPU;
    delete[] resultGPU;

    return 0;
}

int multiplyMatrixWithCPU(int* res, int* matrix1, int* matrix2, unsigned int size) {
    // умножение матриц "в лоб" по алгоритму ijk
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            res[i * size + j] = 0;
            for (int k = 0; k < size; ++k) {
                res[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
            }
        }
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyMatrixWithCUDA(int* res, const int* matrix1, const int* matrix2, unsigned int size)
{
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
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

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, matrix1, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, matrix2, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(startTime, 0);
  
    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(16, 16); // 16 * 16 = 256 threads in block
    dim3 numBlocks(ceil(double(size) / threadsPerBlock.x), ceil(double(size) / threadsPerBlock.y));
    multiplyMatrixKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);

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
    cudaStatus = cudaMemcpy(res, dev_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventElapsedTime(&time, startTime, endTime);

    std::cout << "Time on GPU: " << time << " msec." << std::endl;

Error:
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
