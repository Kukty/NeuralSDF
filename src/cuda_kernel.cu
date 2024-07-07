#include "../include/utils.h"
#include "../include/module.h"

#define BLOCK_SIZE 16
__global__ void add(const float* x, const float* y, float* z, int n) {
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

__global__ void gpu_matrix_mult(const float *a,const float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void run_kernel(const float* a, const float* b, float* c, int m, int n, int k){
        // 调用CUDA内核
    
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, m,n,k);
}

// void DenseLayer::forward_gpu(float *input,float* output)
// {
//     int k = 1;
//     int m = outputshape;
//     int n = inputshape;
//     unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 dimGrid(grid_cols, grid_rows);
//     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//     gpu_matrix_mult<<<dimGrid, dimBlock>>>(weights_gpu, input, output, m,n,k);
//     dim3 blockSize(BLOCK_SIZE);
//     dim3 gridSize((m + blockSize.x - 1) / blockSize.x);
//     add << < dimGrid, dimBlock >> >(output, bias_gpu, output, m);
//     cudaDeviceSynchronize();
// }

// __global__ void sinLayerKernel(float* input, float* output, int size, float w0) {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     if (idx < size) {
//         output[idx] = sin(w0 * input[idx]);
//     }
// }

// void SinLayer::forward_gpu(float *input,float* output)
// {
//     int size = inputshape;
//     int threads = BLOCK_SIZE;
//     int blocks = (size + threads - 1) / threads;

//     sinLayerKernel<<<blocks, threads>>>(input, output, size, w0);
//     cudaDeviceSynchronize();
// }