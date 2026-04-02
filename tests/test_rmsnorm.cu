#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// ==========================================
// 🚀 Core Kernel: RMSNorm
// ==========================================
__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size) {
    // Allocate ultra-fast shared memory visible to all threads in the block
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // 1. Initialize shared memory using the first thread
    if (tid == 0) {
        s_sum = 0.0f;
    }
    // Critical synchronization barrier
    __syncthreads(); 

    if (tid < size) {
        // Step 1: Compute sum of squares using atomic addition to prevent race conditions
        atomicAdd(&s_sum, x[tid] * x[tid]);
    }
    
    // Wait for all threads to finish accumulating
    __syncthreads();

    if (tid < size) {
        // Step 2: Compute normalization coefficient (inverse RMS)
        float ss = 1.0f / sqrtf(s_sum / size + 1e-5f);
        
        // Step 3: Write final normalized and scaled output
        o[tid] = x[tid] * ss * weight[tid];
    }
}

// ==========================================
// 💻 Host Test Code
// ==========================================
int main() {
    std::cout << "[Info] Starting RMSNorm test on RTX 4060..." << std::endl;

    int size = 768; // Simulating a single token's dimension
    size_t bytes = size * sizeof(float);

    float *h_x = new float[size];
    float *h_w = new float[size];
    float *h_o = new float[size];

    // Mock data: inputs = 2.0, weights = 0.5
    // Expected math:
    // 1. Mean of squares: (2.0^2 * 768) / 768 = 4.0
    // 2. Root: sqrt(4.0 + 1e-5) ~= 2.0
    // 3. Inverse RMS (ss) = 1.0 / 2.0 = 0.5
    // 4. Final output o = 2.0 * 0.5 * 0.5 = 0.5
    for (int i = 0; i < size; i++) {
        h_x[i] = 2.0f;
        h_w[i] = 0.5f;
    }

    float *d_x, *d_w, *d_o;
    cudaMalloc((void**)&d_x, bytes);
    cudaMalloc((void**)&d_w, bytes);
    cudaMalloc((void**)&d_o, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes, cudaMemcpyHostToDevice);

    // Launch Kernel: 1 Block, 768 Threads (1-to-1 mapping to dimensions)
    rmsnorm_kernel<<<1, size>>>(d_o, d_x, d_w, size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "[Fatal Error] Kernel failed: " << cudaGetErrorName(err) << std::endl;
        return -1;
    }

    cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost);

    std::cout << "[Result] h_o[0] = " << h_o[0] << " (Expected: ~0.5)" << std::endl;
    std::cout << "[Result] h_o[767] = " << h_o[767] << " (Expected: ~0.5)" << std::endl;

    if (abs(h_o[0] - 0.5f) < 1e-4) {
        std::cout << "[Success] RMSNorm CUDA Kernel is PERFECT!" << std::endl;
    }

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_o);
    delete[] h_x; delete[] h_w; delete[] h_o;

    return 0;
}