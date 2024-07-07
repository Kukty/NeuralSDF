#ifndef UTILS_H
#define UTILS_H
#include<vector>
#include<string>
#include <cuda_runtime.h>
#include"float3_cpu.h"
// #include"neural_network.h"
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
typedef std::vector<std::vector<float>> Matrix;
typedef std::vector<float> Vector;
bool rayIntersectsBox(float3_cpu rayOrigin, float3_cpu rayDirection, float3_cpu boxMin, float3_cpu boxMax, float& travelDistance);
float max_diff(float *res1, float *res2, int n);
int n_zeros(float *a, int n);
void fill_array(float *a, int n);
Vector matrixToVector(Matrix& matrix);
void test_res(float *res1, float *res2, int n);
void print_array(float *a, int n);
void init_zero(float *a, int n);
void set_eq(float *a, float *b, int n);
void kaiming_init(float *w, int n_in, int n_out);
int random_int(int min, int max);
void set_eq(float *a, float *b, int n);
void set_zero(float *a,int n);
bool readSdfTestFile(const std::string &filename, std::vector<Vector> &points, Vector &distances);
bool readSdfArchFile(const std::string &filename, std::vector<std::string> &architecture);
Matrix multiplyMatrices(const Matrix& A, const Matrix& B);
Matrix transposeMatrix(const Matrix& M);
Matrix multiplyMatrices_2(const Matrix& A, const Matrix& B);
float* matrix_to_float_ptr(Matrix &mat);
Vector multiplyMatrice_vector(const Matrix& A, const Vector& B);
void part1(char *argv[]);
void part2(char *argv[]);
// GPU kernel functions 
__global__ void add(const float* x, const float* y, float* z, int n);
// void run_kernel(const float* d_BB, const float* d_CC, float* d_result, int n);
void run_kernel(const float* a, const float* b, float* c, int m, int n, int k);
__global__ void gpu_matrix_mult(const float *a,const float *b, float *c, int m, int n, int k);
__global__ void sinLayerKernel(float* input, float* output, int size, float w0);
void run_test_cuda(void);
void run_test_matrix_multiplication(void);


#endif
