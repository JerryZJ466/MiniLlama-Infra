#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// ==========================================
// 🚀 Core Kernel: Softmax
// ==========================================
// In-place operation on vector x
__global__ void softmax_kernel(float* x, int size) {
    __shared__ float s_max;
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // 1. Find max value (for numerical stability) and initialize sum
    if (tid == 0) {
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        s_max = max_val;
        s_sum = 0.0f; 
    }
    __syncthreads();

    // 2. Concurrently compute exponential and accumulate sum
    if (tid < size) {
        float exp_val = expf(x[tid] - s_max);
        x[tid] = exp_val;
        atomicAdd(&s_sum, exp_val);
    }
    __syncthreads();

    // 3. Concurrently apply final normalization
    if (tid < size) {
        x[tid] = x[tid] / s_sum;
    }
}

// ==========================================
// 💻 Host Test Code
// ==========================================
int main() {
    std::cout << "[Info] Starting Softmax test on RTX 4060..." << std::endl;

    int size = 128; // Simulating sequence length (pos + 1)
    size_t bytes = size * sizeof(float);

    float *h_x = new float[size];
    
    // Mock data: Set first two elements to 10.0, rest to highly negative values
    // Expected math: The probability mass should be equally split (~0.5 each) between the first two tokens.
    for (int i = 0; i < size; i++) h_x[i] = -100.0f; 
    h_x[0] = 10.0f;
    h_x[1] = 10.0f;

    float *d_x;
    cudaMalloc((void**)&d_x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    // Launch Kernel: 1 Block, 128 Threads
    softmax_kernel<<<1, size>>>(d_x, size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "[Fatal Error] Kernel failed: " << cudaGetErrorName(err) << std::endl;
        return -1;
    }

    cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);

    std::cout << "[Result] h_x[0] = " << h_x[0] << " (Expected: ~0.5)" << std::endl;
    std::cout << "[Result] h_x[1] = " << h_x[1] << " (Expected: ~0.5)" << std::endl;
    std::cout << "[Result] h_x[127] = " << h_x[127] << " (Expected: ~0.0)" << std::endl;

    if (abs(h_x[0] - 0.5f) < 1e-4) {
        std::cout << "[Success] Softmax CUDA Kernel is PERFECT!" << std::endl;
    }

    cudaFree(d_x);
    delete[] h_x;

    return 0;
}