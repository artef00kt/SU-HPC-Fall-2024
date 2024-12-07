#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "easyBMP/EasyBMP.h"
#include <string>
#include <time.h> 

// texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

cudaError_t imageInterpCUDA(float* image, float* result, int width, int height);
void imageInterpCPU(float* image, float* result, int width, int height);

void saveImage(float* image, int width, int height, std::string name) {
    BMP Output;
    Output.SetSize(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            Output.SetPixel(j, i, pixel);
        }
    }

    name.append(".bmp");
    Output.WriteToFile(name.c_str());
}

__global__ void interpKernel(cudaTextureObject_t texureObj, float* output, int width, int height)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yj = blockIdx.y * blockDim.y + threadIdx.y;

    output[yj * width * 4 + xi * 2] = tex2D<float>(texureObj, xi, yj);
    output[yj * width * 4 + xi * 2 + 1] = tex2D<float>(texureObj, xi + 0.5, yj);
    output[yj * width * 4 + xi * 2 + width * 2] = tex2D<float>(texureObj, xi, yj + 0.5);
    output[yj * width * 4 + xi * 2 + width * 2 + 1] = tex2D<float>(texureObj, xi + 0.5, yj + 0.5);
}

int main() 
{
    BMP InputImage;
    InputImage.ReadFromFile("wolf.bmp");
    const int width = InputImage.TellWidth();
    const int height = InputImage.TellHeight();

    float* baseImage = (float*)calloc(height * width, sizeof(float));
    float* resultCPU = (float*)calloc(height * 2 * width * 2, sizeof(float));
    float* resultGPU = (float*)calloc(height * 2 * width * 2, sizeof(float));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            baseImage[j * width + i] = (float)floor(
                0.299 * InputImage(i, j)->Red +
                0.587 * InputImage(i, j)->Green +
                0.114 * InputImage(i, j)->Blue
            ); // преобраование к чб и запись в массив
        }
    }

    clock_t startCPU = clock();
    imageInterpCPU(baseImage, resultCPU, width, height);
    clock_t endCPU = clock();
    double timeCPU = endCPU - startCPU;
    std::cout << "Time on CPU: " << timeCPU / CLOCKS_PER_SEC * 1000.0 << " msec." << std::endl;
    saveImage(resultCPU, width * 2, height * 2, "resultCPU");

    imageInterpCUDA(baseImage, resultGPU, width, height);
    saveImage(resultGPU, width * 2, height * 2, "resultGPU");

    delete[] baseImage;
    delete[] resultCPU;
    delete[] resultGPU;
    return 0;
}

void imageInterpCPU(float* image, float* result, int width, int height)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {

            int iIndex, jIndex;
            if (i < width - 1) iIndex = i; else iIndex = -1;
            if (j < height - 1) jIndex = j; else jIndex = -1;
            float f01 = image[j * width + i];
            float f11 = image[j * width + iIndex + 1];
            float f00 = image[(jIndex + 1) * width + i];
            float f10 = image[(jIndex + 1) * width + iIndex + 1];

            float n11 = f01 * 0.5 + f11 * 0.5;
            float n00 = f00 * 0.5 + f01 * 0.5;
            float n10 = f00 * 0.5 * 0.5 + f10 * 0.5 * 0.5 + f01 * 0.5 * 0.5 + f11 * 0.5 * 0.5;

            result[j * width * 4 + i * 2] = f01;
            result[j * width * 4 + i * 2 + 1] = n11;
            result[j * width * 4 + i * 2 + width * 2] = n00;
            result[j * width * 4 + i * 2 + width * 2 + 1] = n10;

        }
    }

}

cudaError_t imageInterpCUDA(float* image, float* result, int width, int height)
{
    float* dev_output = nullptr;
    cudaError_t cudaStatus;

    cudaEvent_t startTime, endTime;
    float time;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    ////////////////

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_arr;
    cudaStatus = cudaMallocArray(&cu_arr, &channelDesc, width, height, cudaArrayTextureGather);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMallocArray failed!"); goto Error; }
    cudaStatus = cudaMemcpyToArray(cu_arr, 0, 0, image, height * width * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpyToArray failed!"); goto Error; }
    // cudaMemcpy2DToArray(cu_arr, 0, 0, image, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_arr;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texureObj;
    cudaCreateTextureObject(&texureObj, &resDesc, &texDesc, nullptr);

    ////////////////


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, width * 2 * height * 2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaEventRecord(startTime, 0);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(16, 16); // 16 * 16 = 256 threads in block
    // dim3 numBlocks(ceil(double(width) / threadsPerBlock.x), ceil(double(height) / threadsPerBlock.y));
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    interpKernel<<<numBlocks, threadsPerBlock>>>(texureObj, dev_output, width, height);

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
    cudaStatus = cudaMemcpy(result, dev_output, width * 2 * height * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventElapsedTime(&time, startTime, endTime);

    std::cout << "Time on GPU: " << time << " msec." << std::endl;

Error:
    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
    cudaFreeArray(cu_arr);
    cudaFree(dev_output);

    return cudaStatus;
}
