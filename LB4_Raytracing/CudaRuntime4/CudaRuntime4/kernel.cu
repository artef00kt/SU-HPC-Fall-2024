#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include "easyBMP/EasyBMP.h"

void saveImage(float3* image, int width, int height, std::string name) {
    BMP Output;
    Output.SetSize(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j].x;
            pixel.Green = image[i * width + j].y;
            pixel.Blue = image[i * width + j].z;
            Output.SetPixel(j, i, pixel);
        }
    }

    name.append(".bmp");
    Output.WriteToFile(name.c_str());
}

// utility
__device__ float3 operator*(const float3& a, const float b) { return { a.x * b, a.y * b, a.z * b }; }
__device__ float operator*(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 operator+(const float3& a, const float3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
__device__ float3 operator-(const float3& a, const float3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
__device__ float3 operator-(const float3& a) { return { -a.x, -a.y, -a.z }; }
__device__ float norm(const float3& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
__device__ float3 normalized(const float3& a) { return a * (1.f / norm(a)); }

struct Material {
    float albedo[3];
    float3 diffuse_color;
    float specular_exponent;
};

struct Sphere {
    float3 position;
    float radius;
    Material material;
};

struct Light {
    float3 position;
    float intensity;
};

__global__ void randSpheres(Sphere* spheres, int num, unsigned  int seed)
{
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (indx < num) {
        curandState state;
        curand_init(seed, indx, 0, &state);
        
        spheres[indx].position = {
            curand_uniform(&state) * 10 - 5.0f,
            curand_uniform(&state) * 10 - 5.0f, 
            -16 + curand_uniform(&state) * 2 - 1.0f
        };
        spheres[indx].radius = 2.0f;

        spheres[indx].material.diffuse_color = { 
            curand_uniform(&state), 
            curand_uniform(&state), 
            curand_uniform(&state)
        };
        spheres[indx].material.specular_exponent = curand_uniform(&state) * 25.0f + 25.0f;// 50.0f;
        spheres[indx].material.albedo[0] = curand_uniform(&state); // 0.6; // diffuse
        spheres[indx].material.albedo[1] = curand_uniform(&state); // 0.3; // specular
        spheres[indx].material.albedo[2] = curand_uniform(&state); // 0.3; // refection
    }
}

__global__ void randLights(Light* lights, int num, unsigned  int seed)
{
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (indx < num) {
        curandState state;
        curand_init(seed, indx, 0, &state);


        lights[indx].position = { 
            50.0f * curand_uniform(&state) - 20.f,
            30.0f * curand_uniform(&state) + 20.0f,
            20 
        };
        lights[indx].intensity = 1.0f + curand_uniform(&state);
    }
}

__device__ float3 reflect(const float3& I, const float3& N) { return I - N * 2.f * (I * N); }

__device__ float2 ray_sphere_intersect(const float3& orig, const float3& dir, const Sphere& s) {
    float3 L = s.position - orig;
    float tca = L * dir;
    float d2 = L * L - tca * tca;
    if (d2 > s.radius * s.radius) return { -1, 0 };
    float thc = std::sqrt(s.radius * s.radius - d2);
    float t0 = tca - thc, t1 = tca + thc;
    if (t0 > .001) return { 1, t0 };
    if (t1 > .001) return { 1, t1 };
    return { -1, 0 };
    // первое значение < 0 пересечения нет, > 0 пересечеение есть
    // второе значение расстояние
}

__device__ bool scene_intersect(const float3& orig, const float3& dir, const Sphere* spheres, int spheres_num, float3& pt, float3& N, Material& material) {
    float spheres_dist = 3.0e+38f;
    for (int i = 0; i < spheres_num; i++) {
        float2 intersect_data = ray_sphere_intersect(orig, dir, spheres[i]);
        bool isIntersect = intersect_data.x > 0;
        float dist_i = intersect_data.y;
        if (isIntersect && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            pt = orig + dir * dist_i; // точка пересечения
            N = normalized((pt - spheres[i].position));
            material = spheres[i].material; // материал сферы с которой пересекаемся
        }
    }
    return spheres_dist < 1000;
}

__device__ float3 cast_ray(const float3& orig, const float3& dir, Sphere* spheres, int spheres_num, Light* lights, int lights_num, int depth) {
    float3 pt, N;
    Material material;

    if (depth > 3 || !scene_intersect(orig, dir, spheres, spheres_num, pt, N, material)) {
        return { 0.3, 0.5, 0.3 };
    }

    // float3 reflect_dir = normalized(reflect(dir, N));
    // float3 reflect_orig = reflect_dir * N < 0 ? pt - N * 1e-3 : pt + N * 1e-3;
    // float3 reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, spheres_num, lights, lights_num, depth + 1);
    // рекурсия не работает, ошибка 719, связана с аппаратным ограничением
    // но вообще отражения работают (создавал дубликат функции, проверял)

    float diffuse_light_intensity = 0, specular_light_intensity = 0;;
    for (int i = 0; i < lights_num; ++i) {
        float3 light_dir = normalized(lights[i].position - pt);

        float light_distance = norm(lights[i].position - pt);
        float3 shadow_orig = light_dir * N < 0 ? pt - N * 1e-3 : pt + N * 1e-3;
        float3 shadow_pt, shadow_N;
        Material tmpmaterial;
        bool isIntersect = scene_intersect(shadow_orig, light_dir, spheres, spheres_num, shadow_pt, shadow_N, tmpmaterial);
        if (isIntersect && norm(shadow_pt - shadow_orig) < light_distance)
            continue;

        diffuse_light_intensity += lights[i].intensity * fmaxf(0.f, light_dir * N);
        specular_light_intensity += powf(fmaxf(0.f, -reflect(-light_dir, N) * dir), material.specular_exponent) * lights[i].intensity;
    }

    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + make_float3(1., 1., 1.) * specular_light_intensity * material.albedo[1]; // + reflect_color * material.albedo[2];
}

__global__ void render(float3 *device_output, int width, int height, Sphere* spheres, int spheres_num, Light* lights, int lights_num)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float fov = 1.05;
    float dir_x = x - width / 2.;
    float dir_y = -y + height / 2.;
    float dir_z = -height / (2. * tan(fov / 2.));
    float3 color = cast_ray({ 0,0,0 }, normalized({ dir_x, dir_y, dir_z }), spheres, spheres_num, lights, lights_num, 0);

    device_output[y * width + x].x = 255 * fminf(fmaxf(0.0, color.x), 1.0);
    device_output[y * width + x].y = 255 * fminf(fmaxf(0.0, color.y), 1.0);
    device_output[y * width + x].z = 255 * fminf(fmaxf(0.0, color.z), 1.0);
}

int main()
{
    int width = 1024;
    int height = 1024;
    int spheres_num = 5;
    int lights_num = 2;

    int seed = 1;

    float3* result = new float3[width * height];;
    float3* device_output;

    cudaError_t cudaStatus;

    cudaEvent_t startTime, endTime;
    float time;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error;}

    // выделение памяти под инициализацию сфер
    Sphere* spheres;
    cudaStatus = cudaMalloc((void**)&spheres, spheres_num * sizeof(Sphere));
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc spheres failed!"); goto Error;}

    // выделение памяти под инициализацию света
    Light* lights;
    cudaStatus = cudaMalloc((void**)&lights, lights_num * sizeof(Light));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc lights failed!"); goto Error; }

    // выделение памяти под рендер
    cudaStatus = cudaMalloc((void**)&device_output, width * height * sizeof(float3));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

    cudaEventRecord(startTime, 0);

    // инициализация сфер
    randSpheres<<<1, spheres_num>>>(spheres, spheres_num, seed);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "randSpheres launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching randSpheres!\n", cudaStatus); goto Error; }

    // инициализация света
    randLights<<<1, lights_num >>>(lights, lights_num, seed);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "randLights launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching randLights!\n", cudaStatus); goto Error; }

    // рендер
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    render<<<numBlocks, threadsPerBlock>>>(device_output, width, height, spheres, spheres_num, lights, lights_num);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "render launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching render!\n", cudaStatus); goto Error;}

    cudaEventRecord(endTime, 0);

    cudaStatus = cudaMemcpy(result, device_output, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}

    cudaEventElapsedTime(&time, startTime, endTime);

    std::cout << "Time on GPU: " << time << " msec." << std::endl;

    saveImage(result, width, height, "result_image");

Error:
    cudaFree(device_output);
    cudaFree(spheres);
    cudaFree(lights);
    delete[] result;
}
