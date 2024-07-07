#include "../include/utils.h"
#include "../include/module.h"
#include "../include/neural_network.h"

#define block_size 8

__global__
void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        out[ind_out] = bias[col];

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = col*n_in + i ;

            out[ind_out] += inp[ind_inp]*weights[ind_weights];
        }
    }
}

// __global__ void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int batch_size, int n_in, int n_out){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     if(idx < batch_size && idy < n_out){
//         float acc = 0.f;
//         for(int i = 0; i < n_in; ++i){
//             acc += inp[idx * n_in + i] * weights[idy * n_out + i];
//         }
//         out[idx * n_out + idy] = acc + bias[idy];
//     }
// }


void DenseLayer::forward_gpu(float *_inp, float *_out,int batch_size){
    inp = _inp;
    out = _out;
    n_out = outputshape;
    n_in = inputshape;
    bs = batch_size;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;
    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights_gpu, bias_gpu, out, bs, inputshape, outputshape);
    cudaDeviceSynchronize();
}

__global__
void linear_backward_gpu(float* gradInput,float *inp, float *weights, float *gradOutput, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;

        for (int i=0; i<n_in; i++){
            ind_inp = row*n_in + i;
            ind_weights = col*n_in + i ;
            gradOutput[ind_inp] += gradInput[ind_out]*weights[ind_weights];
            
        }
    }
}

// __global__ void gpu_matrix_mult(const float *a,const float *b, float *c, int m, int n, int k)
// { 
//     int row = blockIdx.y * blockDim.y + threadIdx.y; 
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0;
//     if( col < k && row < m) 
//     {
//         for(int i = 0; i < n; i++) 
//         {
//             sum += a[row * n + i] * b[i * k + col];
//         }
//         c[row * k + col] = sum;
//     }
// } 

void DenseLayer::backward_gpu(float* gradInput,float* gradOutput,int batch_size){
    
    // dim3 n_blocks(n_block_rows, n_block_cols);
    // dim3 n_threads(block_size, block_size);

    // linear_backward_gpu<<<n_blocks, n_threads>>>(gradInput,inp,cp_weights,gradOutput, bs, n_in, n_out);

    unsigned int grid_rows = (batch_size + block_size - 1) / block_size;
    unsigned int grid_cols = (inputshape + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(gradInput, cp_weights, gradOutput, batch_size,outputshape,inputshape);
    cudaDeviceSynchronize();

    cudaFree(cp_weights);
    cudaFree(out);
}

__global__ void linear_update_gpu(float *inp, float *weights, float *bias, float *gradInput, int bs, int n_in, int n_out, float lr) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
	// int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < bs && col < n_out) {
        const int ind_out = row * n_out + col;

        // m_bias[0] = beta_1 * m_bias[0] + (1 - beta_1) * gradInput[ind_out];
        // v_bias[0] = beta_2 * v_bias[0] + (1 - beta_2) * gradInput[ind_out] * gradInput[ind_out] ; 
        // float m_bias_hat = m_bias[0] / (1 - pow( beta_1,  t + 1));
        // float v_bias_hat = v_bias[0] / (1 - pow( beta_2,  t + 1));

        // optim.m_bias_gpu[0] =  optim.beta_1_gpu[0] *  optim.m_bias_gpu[0] + (1 -  optim.beta_1_gpu[0]) * gradInput[ind_out];
        // optim.v_bias_gpu[0] =  optim.beta_2_gpu[0] *  optim.v_bias_gpu[0] + (1 -  optim.beta_2_gpu[0]) * gradInput[ind_out] * gradInput[ind_out];
        // float m_bias_hat = optim.m_bias_gpu[0] / (1 - pow( optim.beta_1_gpu[0],  optim.t_gpu + 1));
        // float v_bias_hat = optim.v_bias_gpu[0] / (1 - pow( optim.beta_2_gpu[0],  optim.t_gpu + 1));
        // bias updating
        // atomicAdd(&bias[col], -optim.learning_rate * m_bias_hat / (sqrt(v_bias_hat) +  1e-8));

        atomicAdd(&bias[col], -lr * gradInput[ind_out]);

        // weights updating
        for (int i = 0; i < n_in; ++i) {
            const int ind_inp = row * n_in + i;//right 
            const int ind_weights = col * n_in + i;
            
            // m_weights[0] = beta_1 * m_weights[0] + (1 - beta_1) * inp[ind_inp] * gradInput[ind_out];
            // v_weights[0] = beta_2 * v_weights[0] + (1 - beta_2) * inp[ind_inp] * inp[ind_inp] * gradInput[ind_out] * gradInput[ind_out];
            // float m_weights_hat = m_weights[0] / (1 - pow(beta_1, t + 1));
            // float v_weights_hat = v_weights[0] / (1 - pow(beta_2, t + 1));

            // optim.m_weights_gpu[0] = optim.beta_1_gpu[0] * optim.m_weights_gpu[0] + (1 - optim.beta_1_gpu[0]) * inp[ind_inp] * gradInput[ind_out];
            // optim.v_weights_gpu[0] = optim.beta_2_gpu[0] * optim.v_weights_gpu[0] + (1 - optim.beta_2_gpu[0]) * inp[ind_inp] * inp[ind_inp] * gradInput[ind_out] * gradInput[ind_out];
            // float m_weights_hat = optim.m_weights_gpu[0] / (1 - pow(optim.beta_1_gpu[0], optim.t_gpu + 1));
            // float v_weights_hat = optim.v_weights_gpu[0] / (1 - pow(optim.beta_2_gpu[0], optim.t_gpu + 1));
            // atomicAdd(&weights[ind_weights], -optim.learning_rate * m_weights_hat / (sqrt(v_weights_hat) +  1e-8));

            atomicAdd(&weights[ind_weights], -lr * inp[ind_inp] * gradInput[ind_out]);
        }
    }

}

void DenseLayer::update_gpu(float* gradInput,OptimizerAdam& optim,int batch_size)
{
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    
    set_eq(cp_weights, weights_gpu, sz_weights);
    n_block_rows = (batch_size + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;
    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights_gpu, bias_gpu, gradInput, batch_size, n_in, n_out, optim.learning_rate);
    optim.t_gpu+=1;
    cudaDeviceSynchronize();
}

__global__
void sin_forward_gpu(float *inp, float *out,int w0, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        out[ind] = sin(w0 * inp[ind]);
    }
}


void SinLayer::forward_gpu(float *_inp, float *_out,int batch_size){

    n_blocks = (batch_size*outputshape + block_size - 1) / block_size;
    inp = _inp;
    out = _out;

    sin_forward_gpu<<<n_blocks, block_size>>>(inp, out,w0, batch_size*outputshape);
    cudaDeviceSynchronize();
}

void SinLayer::update_gpu(float* gradInput,OptimizerAdam& optim,int batch_size)
{
}

__global__
void sin_backward_gpu(float *inp,float* gradInput, float *gradOutput,int w0, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        gradOutput[ind] = gradInput[ind] * w0 * cos(w0 * inp[ind]);
    }
}

void SinLayer::backward_gpu(float* gradInput,float* gradOutput,int batch_size){
    int sz_out = batch_size*outputshape;
    n_blocks = (batch_size*outputshape + block_size - 1) / block_size;
    sin_backward_gpu<<<n_blocks, block_size>>>(inp,gradInput,gradOutput,w0, sz_out);
    cudaDeviceSynchronize();

    cudaFree(out);
}

__global__
void mse_backward_gpu(float* predict, float* target, float* gradOutput, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        gradOutput[ind] = fdividef(2*(predict[ind]-target[ind]), sz_out);
    }
}

void MSE::backward_gpu(float* predict, float* target, float* gradOutput,int batch_size)
{

    int n_blocks = (batch_size + block_size - 1) / block_size;
    mse_backward_gpu<<<n_blocks, block_size>>>(predict, target,gradOutput, batch_size);
    cudaDeviceSynchronize();
}



// FOR GPU Rending

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}
__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

__device__ __host__ float3 operator*(const float3 &a, const float &b) {

  return make_float3(a.x*b, a.y*b, a.z*b);

}

__device__ __host__ float3 operator/(const float3 &a, const float &b) {

  return make_float3(a.x/b, a.y/b, a.z/b);

}

__device__ __host__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

__device__ __host__  bool rayIntersectsBox(float3 rayOrigin, float3 rayDirection, float3 boxMin, float3 boxMax, float &travelDistance) {
    float tmin = 0.0f;
    float tmax = 100000000000.0;

    
    if (fabsf(rayDirection.x) < 1e-6f) {
        if (rayOrigin.x < boxMin.x || rayOrigin.x > boxMax.x) {
            return false;
        }
    } else {
        float ood = 1.0f / rayDirection.x;
        float t1 = (boxMin.x - rayOrigin.x) * ood;
        float t2 = (boxMax.x - rayOrigin.x) * ood;
        if (t1 > t2) {
            float temp = t1; t1 = t2; t2 = temp;
        }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) {
            return false;
        }
    }
    if (fabsf(rayDirection.y) < 1e-6f) {
            if (rayOrigin.y < boxMin.y || rayOrigin.y > boxMax.y) {
                return false;
            }
        } else {
            float ood = 1.0f / rayDirection.y;
            float t1 = (boxMin.y - rayOrigin.y) * ood;
            float t2 = (boxMax.y - rayOrigin.y) * ood;
            if (t1 > t2) {
                float temp = t1; t1 = t2; t2 = temp;
            }
            tmin = fmaxf(tmin, t1);
            tmax = fminf(tmax, t2);
            if (tmin > tmax) {
                return false;
            }
        }
    if (fabsf(rayDirection.z) < 1e-6f) {
            if (rayOrigin.z < boxMin.z || rayOrigin.z > boxMax.z) {
                return false;
            }
        } else {
            float ood = 1.0f / rayDirection.z;
            float t1 = (boxMin.z - rayOrigin.z) * ood;
            float t2 = (boxMax.z - rayOrigin.z) * ood;
            if (t1 > t2) {
                float temp = t1; t1 = t2; t2 = temp;
            }
            tmin = fmaxf(tmin, t1);
            tmax = fminf(tmax, t2);
            if (tmin > tmax) {
                return false;
            }
        }

    travelDistance = tmin;
    return true;
}
__device__ __host__  float* float3_to_ptr(float3& vec) {
    // 注意：这里直接取地址转换为float*，因为vec已经是非const的
    return reinterpret_cast<float*>(&vec);
}
__device__ __host__  float length(const float3 v) {
    return sqrt(v.x * v.x + v.y+v.y + v.z*v.z );
}


// __device__ __host__  float DistanceEvaluation(NeuralNetwork network_gpu, float3 v)
// {
//     float *out;
//     float* input = float3_to_ptr(v);
//     network_gpu.forward_gpu(input,1);
//     out = network_gpu.layers.back()->out;
//     float distance = out[0];
//     return distance;
// }

// __device__ __host__  float3 EstimateNormal(NeuralNetwork network,float3 z)
// {
//     float eps = 0.0000001;
//     float3 z1 = z + make_float3(eps, 0, 0);
//     float3 z2 = z - make_float3(eps, 0, 0);
//     float3 z3 = z + make_float3(0, eps, 0);
//     float3 z4 = z - make_float3(0, eps, 0);
//     float3 z5 = z + make_float3(0, 0, eps);
//     float3 z6 = z - make_float3(0, 0, eps);
//     float dx = DistanceEvaluation(network,z1) - DistanceEvaluation(network,z2);
//     float dy = DistanceEvaluation(network,z3) - DistanceEvaluation(network,z4);
//     float dz = DistanceEvaluation(network,z5) - DistanceEvaluation(network,z6);
//     return normalize(make_float3(dx, dy, dz) / (2.0f * eps));

// }


// __global__ void kernel_raytrace(float* image,float3 lookDirection,float3 right,float3 up,float3 pos, int image_width,int image_height,float fovTanHalf,float aspectRatio,float3 light_dir,NeuralNetwork network_gpu)
// {
//     size_t widthIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t heightIndex = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t idx = (heightIndex * image_width + widthIndex) * 3;
//     if (widthIndex < image_width && heightIndex < image_height) {
//         float normX = (2.0f * widthIndex + 1) / image_width - 1;
//         float normY = (2.0f * heightIndex + 1) / image_height - 1;
//         float3 direction = normalize(lookDirection + right * normX * fovTanHalf * aspectRatio + up * normY * fovTanHalf);
//         float init_t = 0 ;
//         bool flag = rayIntersectsBox(pos,direction,make_float3(-1,-1,-1),make_float3(1,1,1),init_t);
//         if (flag == false){
//             return;
//         }
//         float3 v = pos+  direction * init_t;
//         float3 col = make_float3(0.0,0.0,0.0);

//         float3 intersection = make_float3(0.0,0.0,0.0);
//         float3 pixelColor = make_float3(0.0,0.0,0.0);
        
//         do{
//             float distance = DistanceEvaluation(network_gpu,v);
            
//             if (abs(distance) < 0.0001)
//                 {
//                     col = make_float3(1.0,1.0,1.0);
//                     intersection = v;
//                     break;
//                 }
//                 v = v + direction * distance;
//             }while (length(v)<1.1);
//         if (length(intersection) > 0) 
//             {
//                     float3 surfaceNormal =EstimateNormal(network_gpu,intersection);
                    
                    
//                     col = col * max(0.1f, dot(surfaceNormal,light_dir));
                    
//                     pixelColor = col;
//                     // std::cout << pixelColor.x <<std::endl;
                
//                 image[idx] = pixelColor.x;
//                 image[idx + 1] = pixelColor.y;
//                 image[idx + 2] = pixelColor.z;
//             }
//     }
// }
// void run_gpu_raytrace(float* image,float3 lookDirection,float3 right,float3 up,float3 pos, int image_width,int image_height,float fovTanHalf,float aspectRatio,float3 light_dir,NeuralNetwork network_gpu)
// {
//     // float3 q = lookDirection + right;
//     int n_block_rows = (image_width + block_size - 1) / block_size;
//     int n_block_cols = (image_height + block_size - 1) / block_size;
//     dim3 n_blocks(n_block_rows, n_block_cols);
//     dim3 n_threads(block_size, block_size);
//     kernel_raytrace<<<n_blocks,n_threads>>>(image,lookDirection,right,up,pos,image_height,image_height,fovTanHalf,aspectRatio,light_dir,network_gpu);
// }
// // __device__ 