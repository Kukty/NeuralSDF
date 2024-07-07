#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../include/utils.h"
#include <fstream>
#include <cmath>


float max_diff(float *res1, float *res2, int n){
    float diff, r = 0;

    for (int i=0; i<n; i++){
        diff = abs(res1[i]-res2[i]);
        r = (r < diff) ? diff : r;
    }

    return r;
}

Vector matrixToVector(Matrix& matrix) {
    int rows = matrix.size(); // 获取矩阵的行数
    int cols = matrix[0].size(); // 获取矩阵的第一行的列数（假设所有行都有相同数量的列）
    int size = rows * cols;     // 矩阵的元素个数

    std::vector<float> vectorData(size); // 创建可容纳矩阵所有元素的一维数组

    // 将矩阵元素拷贝到一维数组中
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            vectorData[i * cols + j] = matrix[i][j];
        }
    }

    return vectorData;
}
float* matrix_to_float_ptr(Matrix &mat) {
    int size = mat.size() * mat[0].size();
    float* ptr = new float[size];
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            ptr[i * mat[0].size() + j] = mat[i][j];
        }
    }
    return ptr;
}

bool rayIntersectsBox(float3_cpu rayOrigin, float3_cpu rayDirection, float3_cpu boxMin, float3_cpu boxMax, float& travelDistance) {
    float tmin = 0.0f;
    float tmax = std::numeric_limits<float>::max();

    for (int i = 0; i < 3; ++i) {
        if (std::abs(rayDirection[i]) < 1e-6) {
            if (rayOrigin[i] < boxMin[i] || rayOrigin[i] > boxMax[i]) {
                return false;
            }
        } else {
            float ood = 1.0f / rayDirection[i];
            float t1 = (boxMin[i] - rayOrigin[i]) * ood;
            float t2 = (boxMax[i] - rayOrigin[i]) * ood;
            if (t1 > t2) {
                std::swap(t1, t2);
            }
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) {
                return false;
            }
        }
    }

    travelDistance = tmin;
    return true;
}
void set_eq(float *a, float *b, int n){
    for (int i=0; i<n; i++){
        a[i] = b[i];
    }
}
void set_zero(float*a, int n)
{
    for (int i=0; i<n; i++){
        a[i] = 0;
    }
}

int n_zeros(float *a, int n){
    int r = 0;

    for (int i=0; i<n; i++){
        r += (!a[i]);
    }
    
    return r;
}


void fill_array(float *a, int n){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, 1.0f); 

    for (int i=0; i<n; i++){
        a[i] = dist(gen);
    }
}


void test_res(float *res1, float *res2, int n){
    int n_res1_zeros = n_zeros(res1, n), n_res2_zeros = n_zeros(res2, n);
    float mx = max_diff(res1, res2, n);

    std::cout << "Number of zeros of res1: " << n_res1_zeros << std::endl;
    std::cout << "Number of zeros of res2: " << n_res2_zeros << std::endl;
    std::cout << "Maximum difference: " << mx << std::endl;
    std::cout << "*********" << std::endl;
}


void print_array(float *a, int n){
    for (int i=0; i<n; i++){
        std::cout << a[i] << std::endl;
    }
    std::cout << "*********" << std::endl;
}


void init_zero(float *a, int n){
    for (int i=0; i<n; i++){
        a[i] = 0.0f;
    }
}






int random_int(int min, int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// Function to perform matrix multiplication


// Function to transpose a matrix
Matrix transposeMatrix(const Matrix& M) {
    size_t rows = M.size();
    size_t cols = M[0].size();
    
    Matrix T(cols, std::vector<float>(rows, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            T[j][i] = M[i][j];
        }
    }
    
    return T;
}

Matrix multiplyMatrices(const Matrix& A, const Matrix& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    Matrix Bt = transposeMatrix(B);
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrices cannot be multiplied due to incompatible dimensions.");
    }

    Matrix C(rowsA, std::vector<float>(colsB, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * Bt[j][k];
            }
        }
    }

    return C;
}

Matrix multiplyMatrices_2(const Matrix& A, const Matrix& B){
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();
    Matrix C(rowsA, std::vector<float>(colsB, 0.0));
    #pragma omp parallel for 
    for (size_t i = 0; i < rowsA; i++) 
        for (size_t k = 0; k < colsA; k++)
            for (size_t j = 0; j < colsB; j++) 
                C[i][j] += A[i][k] * B[k][j];
            
    return C;
}

Vector multiplyMatrice_vector(const Matrix& A, const Vector& B){
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    Vector C(rowsA);
    #pragma omp parallel for 
    for (size_t i = 0; i < rowsA; i++) 
        for (size_t k = 0; k < colsA; k++)
                C[i] += A[i][k] * B[k];
    return C;
}

bool readSdfTestFile(const std::string &filename, Matrix &points, Vector &distances) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    // Read the number of points N
    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    if (N <= 0) {
        std::cerr << "Error: Invalid number of points" << std::endl;
        return false;
    }
    
    // Read the point coordinates
    points.resize(N, Vector(3));
    for (int i = 0; i < N; ++i) {
        file.read(reinterpret_cast<char*>(points[i].data()), 3 * sizeof(float));
    }

    // Read the reference distances
    distances.resize(N);
    file.read(reinterpret_cast<char*>(distances.data()), N * sizeof(float));

    file.close();
    return true;
}

bool readSdfArchFile(const std::string &filename, std::vector<std::string> &architecture) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    // Read the number of points N
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            architecture.push_back(line);
        }
        file.close();
    }
    return true;
}

void run_test_cuda(void){
    // Matrix AA = {
    //     {1, 2},
    //     {3, 4}
    // };
    
    // Vector BB = {5, 6};
    // Vector CC = {6, 6};
    // // Vector CC = multiplyMatrice_vector(AA,BB);
    // // for (const auto& elem : CC) {
    // //         std::cout << elem << ' ';
    // //     }
    // // 在GPU上分配内存
    // float* d_BB;
    // float* d_CC;
    // float* d_result;
    // cudaMalloc((void**)&d_BB, BB.size() * sizeof(float));
    // cudaMalloc((void**)&d_CC, CC.size() * sizeof(float));
    // cudaMalloc((void**)&d_result, BB.size() * sizeof(float));

    // // 将数据从主机复制到设备
    // cudaMemcpy(d_BB, BB.data(), BB.size() * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_CC, CC.data(), CC.size() * sizeof(float), cudaMemcpyHostToDevice);

    // int n = BB.size();
    // run_kernel(d_BB,d_CC,d_result,n);
    

    // // 将结果从设备复制到主机
    // Vector result(BB.size());
    // cudaMemcpy(result.data(), d_result, BB.size() * sizeof(float), cudaMemcpyDeviceToHost);


    // // 输出结果
    // for (const auto& elem : result) {
    //     std::cout << elem << ' ';
    // }
    // std::cout << std::endl;

    // // 释放GPU内存
    // cudaFree(d_BB);
    // cudaFree(d_CC);
    // cudaFree(d_result);
        // run_test_matrix_multiplication();

//    // ---------------------------------------------n x n * n x 1 = > n x 1 --------------------
    // Matrix AA = {
    //     {1, 2},
    //     {3, 4}
    // };
    // int size =2 ;
    // Vector BB = {5, 6};
    // Vector CC = multiplyMatrice_vector(AA,BB);
    
    // for (const auto& elem : CC) {
    //         std::cout << elem << ' ';
    //     }
    // std::cout <<std::endl;
    // float *a = matrix_to_float_ptr(AA);
    // float *c = new float[size * 1];
    // float *d_b;
    // float *d_a;
    // float* d_c;
    // cudaMalloc((void**)&d_a, size * size * sizeof(float));
    // cudaMalloc((void**)&d_b, size * 1 * sizeof(float));
    // cudaMalloc((void**)&d_c, size * 1 * sizeof(float));
    
    // cudaMemcpy(d_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, BB.data(), size * 1 * sizeof(float), cudaMemcpyHostToDevice);
    // run_kernel(d_a,d_b,d_c,size,size,1);
    // cudaMemcpy(c, d_c, size * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0;i<size;i++)
    // {
    //     std::cout<<c[i]<<' ';
    // }
    // std::cout<<std::endl;
    //

}

void run_test_matrix_multiplication(void)
{
    // -----------------------------matrix multiple------------------------------------------------

    const int size = 768;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 2.0);

    // 生成随机矩阵A和B
    Matrix A(size, std::vector<float>(size));
    Matrix B(size, std::vector<float>(size));

    // for (int i = 0; i < size; i++) {
    //     for (int j = 0; j < size; j++) {
    //         A[i][j] = dis(gen);
    //         B[i][j] = dis(gen);
    //     }
    // }
    auto start = std::chrono::steady_clock::now();
    Matrix C = multiplyMatrices(A, B);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Standard matrix multiplication took " << elapsed_seconds.count() << " seconds." << std::endl;
    // for (const auto& row : C) {
    //     for (const auto& elem : row) {
    //         std::cout << elem << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    start = std::chrono::steady_clock::now();
    Matrix D = multiplyMatrices_2(A,B);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Optimized matrix multiplication took " << elapsed_seconds.count() << " seconds." << std::endl;

    // ----gpu-----
    start = std::chrono::steady_clock::now();
    float *a = matrix_to_float_ptr(A);
    float *b = matrix_to_float_ptr(B);
    float *c = new float[size * size];
    // Matrix CC(size, std::vector<float>(size));
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, size * size * sizeof(float));
    cudaMalloc((void**)&d_b, size * size * sizeof(float));
    cudaMalloc((void**)&d_c, size * size * sizeof(float));

    cudaMemcpy(d_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Transfer data from cpu to gpu took " << elapsed_seconds.count() << " seconds." << std::endl;
    start = std::chrono::steady_clock::now();
    run_kernel(d_a,d_b,d_c,size,size,size);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "GPU matrix multiplication took " << elapsed_seconds.count() << " seconds." << std::endl;
    cudaMemcpy(c, d_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = c[i * size + j];
        }
    }
    // for (const auto& row : C) {
    //     for (const auto& elem : row) {
    //         std::cout << elem << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
