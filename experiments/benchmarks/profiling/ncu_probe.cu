#include <cuda_runtime.h>

#include <cstdio>

__global__ void streamfm_ncu_probe(float* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = values[index] * 1.0001f + 0.25f;
    }
}

int main() {
    constexpr int count = 1 << 20;
    float* values = nullptr;
    cudaError_t status = cudaMalloc(&values, count * sizeof(float));
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return 20;
    }

    streamfm_ncu_probe<<<(count + 255) / 256, 256>>>(values, count);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    cudaFree(values);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "probe kernel failed: %s\n", cudaGetErrorString(status));
        return 21;
    }

    std::puts("streamfm_ncu_probe_ok");
    return 0;
}
