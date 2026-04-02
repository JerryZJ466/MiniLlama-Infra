#include <cuda_runtime.h>
#include <iostream>

// ==========================================
// 🚀 1. Core Kernel: Concurrent Matrix Multiplication on GPU
// ==========================================
// Computes W(d, n) * x(n,) -> xout(d,)
__global__ void matmul_kernel(float* xout, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

int main() {
    std::cout << "[Info] Initializing Matrix Multiplication on RTX 4060..." << std::endl;

    // Simulating a layer in the Llama model: input dim n=288, output dim d=768
    int n = 288;
    int d = 768;
    size_t size_w = n * d * sizeof(float); 
    size_t size_x = n * sizeof(float);     
    size_t size_xout = d * sizeof(float);  

    // 1. Allocate Host memory and initialize mock data
    float *h_w = new float[n * d];
    float *h_x = new float[n];
    float *h_xout = new float[d];

    // Initialize W with 1.0 and x with 2.0
    for (int i = 0; i < n * d; i++) h_w[i] = 1.0f;
    for (int i = 0; i < n; i++) h_x[i] = 2.0f;

    // 2. Allocate Device memory (VRAM)
    float *d_w, *d_x, *d_xout;
    cudaMalloc((void**)&d_w, size_w);
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_xout, size_xout);

    // 3. [H2D] Transfer data to GPU
    cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

    // 4. [Compute] Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "[Info] Launching Kernel with " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block." << std::endl;
    
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_xout, d_x, d_w, n, d);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("[Fatal Error] Launch Error Code: %d\n", (int)launch_err);
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("[Fatal Error] Sync Error Code: %d\n", (int)sync_err);
    }

    // 5. [D2H] Transfer results back to Host
    cudaMemcpy(h_xout, d_xout, size_xout, cudaMemcpyDeviceToHost);

    // 6. Verify Results
    // Mathematical logic: W=1.0, x=2.0. Each row's dot product is 288 * 1.0 * 2.0 = 576.0.
    std::cout << "[Result] xout[0] = " << h_xout[0] << " (Expected: 576)" << std::endl;
    std::cout << "[Result] xout[767] = " << h_xout[767] << " (Expected: 576)" << std::endl;

    if (h_xout[0] == 576.0f) {
        std::cout << "[Success] CUDA Matrix Multiplication is PERFECT!" << std::endl;
    }

    // 7. Resource Cleanup
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xout);
    delete[] h_w; delete[] h_x; delete[] h_xout;

    return 0;
}