// module.cpp
#include "../include/module.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
DenseLayer::DenseLayer(const Matrix& w, const Vector& b,int inshape,int outshape) : Module(inshape, outshape),weights(w), bias(b){
    copy_weights = Matrix(weights.size(),Vector(weights[0].size(),0));
}
// void Module::forward_gpu(float *input, float* output) {}
Vector DenseLayer::forward(const Vector& input) const {
    Vector output = multiplyMatrice_vector(weights,input);
    for (size_t i = 0; i < weights.size(); ++i) {
        output[i] += bias[i];
    }
    return output;
}

Matrix DenseLayer::forward_batch(Matrix& input){
    int batch_size = input.size();
    // Matrix newinput = Matrix(batch_size,Vector(inputshape,0));
    saved_input = Matrix(batch_size, Vector(inputshape, 0.0));

    Matrix output(batch_size, Vector(outputshape, 0.0));
    for (size_t i =0; i < batch_size ; ++i){
        saved_input[i] = input[i];
        output[i] = forward(input[i]);
    }
    return output;
}

void DenseLayer:: update(Matrix& gradInput,OptimizerAdam& optim)
{
    for(int i =0;i<weights.size();i++)
    {
        for(int j = 0;j < weights[0].size(); j++)
        {
            copy_weights[i][j] = weights[i][j];
        }
    }
    int batch_size =gradInput.size();
    #pragma omp parallel for 
    for (int i=0; i<batch_size; i++)
    {
        for(int k=0;k<outputshape;k++)
        {   

            bias[k] -= optim.learning_rate * gradInput[i][k];


            // optim.m_bias =  optim.beta_1 *  optim.m_bias + (1 -  optim.beta_1) * gradInput[i][k];
            // optim.v_bias =  optim.beta_2 *  optim.v_bias + (1 -  optim.beta_2) * gradInput[i][k] * gradInput[i][k];
            // float m_bias_hat = optim.m_bias / (1 - std::pow( optim.beta_1,  optim.t + 1));
            // float v_bias_hat = optim.v_bias / (1 - std::pow( optim.beta_2,  optim.t + 1));
            // bias[k] -= optim.learning_rate * m_bias_hat / (std::sqrt(v_bias_hat) + optim.eps);
            // std::cout << gradInput[i][k] <<std::endl;
            for(int j = 0; j < inputshape;j++)
            {

                // optim.m_weights = optim.beta_1 * optim.m_weights + (1 - optim.beta_1) * saved_input[i][j] * gradInput[i][k];
                // optim.v_weights = optim.beta_2 * optim.v_weights + (1 - optim.beta_2) * saved_input[i][j] * saved_input[i][j] * gradInput[i][k] * gradInput[i][k];
                // float m_weights_hat = optim.m_weights / (1 - std::pow(optim.beta_1, optim.t + 1));
                // float v_weights_hat = optim.v_weights / (1 - std::pow(optim.beta_2, optim.t + 1));
                // weights[k][j] -= optim.learning_rate * m_weights_hat / (std::sqrt(v_weights_hat) + optim.eps);
                
                weights[k][j] -= optim.learning_rate * saved_input[i][j] * gradInput[i][k];
            }
        }
    }

}

Vector DenseLayer::backward(const Vector& dLdY) {
    Vector dLdX(inputshape, 0.0);
    for (size_t j = 0; j < outputshape; ++j) 
    {
        for (size_t i = 0; i < inputshape; ++i) 
            {
                dLdX[i] += dLdY[j] * copy_weights[j][i];
            }

    }
    return dLdX;
}

Matrix DenseLayer::backward_batch(const Matrix& gradInput)  
{
    return multiplyMatrices_2(gradInput,copy_weights);
}

int DenseLayer::getInputSize() const {
    return weights[0].size();
}

int DenseLayer::getOutputSize() const {
    return weights.size();
}

SinLayer::SinLayer(int inshape,int outshape):Module(inshape, outshape){}

Vector SinLayer::forward(const Vector& input) const {
    Vector output = input;
    for (auto& val : output) {
        val = std::sin(w0 * val);
    }
    return output;
}

Matrix SinLayer::forward_batch( Matrix& input) {
    int batch_size = input.size();
    // Matrix newinput = Matrix(batch_size,Vector(inputshape,0));
    saved_input = Matrix(batch_size, Vector(inputshape, 0.0));
    Matrix output(batch_size, Vector(outputshape, 0.0));
    for (size_t i =0; i < batch_size ; ++i){
        saved_input[i] = input[i];
        output[i] = forward(input[i]);
    }
    return output;
}


Vector SinLayer::backward(const Vector& dL_dy, Vector& input) {
    Vector dL_dinput(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        dL_dinput[i] = dL_dy[i] * w0 * std::cos(w0 * input[i]);
    }

    return dL_dinput;
}

Matrix SinLayer::backward_batch(const Matrix& gradInput) 
{
    int batch_size = gradInput.size();
    Matrix gradOutput(batch_size, Vector(inputshape, 0.0));
    for (size_t i = 0; i < batch_size; ++i) {
        gradOutput[i] = backward(gradInput[i],saved_input[i]);
    }
    return gradOutput;
}

void SinLayer::update(Matrix& gradInput,OptimizerAdam& optim)
{
    0;
}

int SinLayer::getInputSize() const {
    return 0; // No specific input size for SinLayer
}

int SinLayer::getOutputSize() const {
    return 0; // No specific output size for SinLayer
}

Dataloader::Dataloader(Matrix dataset,Vector targets,int batch_size,bool shuffle):
dataset(dataset),targets(targets),batch_size(batch_size),shuffle(shuffle){
    assert(dataset.size() == targets.size());
    len = dataset.size() / batch_size;

    std::cout<< "dataloader created, batch_szie = " << batch_size << ", dataloader len = " << len << std::endl;
    // Drop last
}

BatchResult Dataloader::Get_batch(int index)
{
    assert(index < len);
    if (index == 0 )
    {
        if (shuffle) 
        {
            unsigned seed = 123456;
            std::mt19937 gen(seed);
            std::shuffle(dataset.begin(), dataset.end(), gen);
            std::shuffle(targets.begin(), targets.end(), gen);
        }
    }

    BatchResult result;
    int start = index * batch_size;
    int end = (index+1) * batch_size;
    result.input.assign(dataset.begin() + start ,dataset.begin() + end);
    result.target.assign(targets.begin() + start ,targets.begin() + end);
    return result;
}


float MSE::forward(const Vector& predict,const Vector& target){
    float res = 0;
    int batch_size = predict.size();
    for(int i=0;i<batch_size;i++)
    {
        res += (target[i] - predict[i]) * (target[i] - predict[i]) / batch_size;
    }
    return res;
}

Matrix MSE::backward(const Matrix& predict,const Vector& target){
    
    int batch_size = predict.size();
    Matrix grad(batch_size,Vector(1,0));
    for(int i=0;i<batch_size;i++)
    {
        grad[i][0] = 2.0 * (predict[i][0] - target[i] )/ batch_size;
    }
    return grad;
}

OptimizerAdam::OptimizerAdam(float _lr, float _beta_1, float _beta_2, float _eps)
    : learning_rate(_lr), beta_1(_beta_1), beta_2(_beta_2), eps(_eps),
      m_bias(0.0f), v_bias(0.0f), m_weights(0.0f), v_weights(0.0f), t(0) {}

void OptimizerAdam::to_gpu(void)
{
    // cudaMalloc((void**)&beta_1_gpu, 1 * sizeof(float));
    cudaMallocManaged(&beta_1_gpu, 1 * sizeof(float));
    cudaMemcpy(beta_1_gpu, &beta_1, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&beta_2_gpu, 1 * sizeof(float));
    cudaMallocManaged(&beta_2_gpu, 1 * sizeof(float));
    cudaMemcpy(beta_2_gpu, &beta_2, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&eps_gpu, 1 * sizeof(float));
    cudaMallocManaged(&eps_gpu, 1 * sizeof(float));
    cudaMemcpy(eps_gpu, &eps, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&m_bias_gpu, 1 * sizeof(float));
    cudaMallocManaged(&m_bias_gpu, 1 * sizeof(float));
    cudaMemcpy(m_bias_gpu, &m_bias, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&v_bias_gpu, 1 * sizeof(float));
    cudaMallocManaged(&v_bias_gpu, 1 * sizeof(float));
    cudaMemcpy(v_bias_gpu, &v_bias, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&m_weights_gpu, 1 * sizeof(float));
    cudaMallocManaged(&m_weights_gpu, 1 * sizeof(float));
    cudaMemcpy(m_weights_gpu, &m_weights, sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&v_weights_gpu, 1 * sizeof(float));
    cudaMallocManaged(&v_weights_gpu, 1 * sizeof(float));
    cudaMemcpy(v_weights_gpu, &v_weights, sizeof(float), cudaMemcpyHostToDevice);
    t_gpu = 0 ;
    // cudaMalloc((void**)&t_gpu, 1 * sizeof(int));
    // cudaMallocManaged(&t_gpu, 1 * sizeof(int));
    // cudaMemcpy(t_gpu, &t, sizeof(int), cudaMemcpyHostToDevice);
}