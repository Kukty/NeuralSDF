// neural_network.cpp
#include "../include/neural_network.h"
#include "../include/module.h"
#include "../include/utils.h"
#include <sstream>
#include <fstream>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<std::string>& architecture) {
    for (const auto& line : architecture) {
        std::istringstream iss(line);
        std::string layerType;
        std::string inputShapeStr, outputShapeStr;

        iss >> layerType >> inputShapeStr >> outputShapeStr;
        int inputShape = 0;
        int outputShape = 0;
        size_t inputPos = line.find("input shape");
        size_t outputPos = line.find("output shape");

        // 提取 input shape 和 output shape 的值
        

        // 提取 input shape
        for (size_t i = inputPos + 13; i < line.length(); i++) {
            if (line[i] == ')') {
                break;
            }
            inputShape = inputShape * 10 + (line[i] - '0');
        }

        // 提取 output shape
        for (size_t i = outputPos + 14; i < line.length(); i++) {
            if (line[i] == ')') {
                break;
            }
            outputShape = outputShape * 10 + (line[i] - '0');
        }

        if (layerType == "Dense") {
            Matrix weights(outputShape, Vector(inputShape,0.0));
            //Kaiming init:
            size_t rows = weights.size();
            size_t cols = weights[0].size();
            std::random_device rd;
            std::mt19937 gen(rd());
            float omega_0 = 30.0;
            std::uniform_real_distribution<> dis(-std::sqrt(6.0 / inputShape) / omega_0, std::sqrt(6.0 / inputShape) / omega_0);
            float variance = 2.0f / (rows + cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    weights[i][j]= dis(gen);
                }
            } 
            Vector bias(outputShape,0.0);


            layers.push_back(new DenseLayer(weights, bias,inputShape,outputShape));
        } else if (layerType == "Sin") {
            layers.push_back(new SinLayer(inputShape,outputShape));
        }
    }
}
NeuralNetwork::NeuralNetwork(const std::vector<std::string>& architecture,const std::string &weightsfilename){
    std::ifstream file(weightsfilename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << weightsfilename << std::endl;
    }
    for (const auto& line : architecture) {
        std::istringstream iss(line);
        std::string layerType;
        std::string inputShapeStr, outputShapeStr;

        iss >> layerType >> inputShapeStr >> outputShapeStr;
        int inputShape = 0;
        int outputShape = 0;
        size_t inputPos = line.find("input shape");
        size_t outputPos = line.find("output shape");
        for (size_t i = inputPos + 13; i < line.length(); i++) {
            if (line[i] == ')') {
                break;
            }
            inputShape = inputShape * 10 + (line[i] - '0');
        }

        // 提取 output shape
        for (size_t i = outputPos + 14; i < line.length(); i++) {
            if (line[i] == ')') {
                break;
            }
            outputShape = outputShape * 10 + (line[i] - '0');
        }
        if (layerType == "Dense") {

            Matrix weights(outputShape, Vector(inputShape));
            Vector bias(outputShape);
            for(size_t i=0;i<weights.size();++i){
                file.read(reinterpret_cast<char*>(weights[i].data()), weights[i].size() * sizeof(float));
            }
                // 读取偏置
            file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
            // Initialize weights and bias (you can read them from a file)

            layers.push_back(new DenseLayer(weights, bias,inputShape,outputShape));
        } else if (layerType == "Sin") {
            layers.push_back(new SinLayer(inputShape,outputShape));
        }

    }
}
Vector NeuralNetwork::forward(Vector& input) const {

    Vector output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
        // std::cout<< output[0]<<std::endl;
    }
    return output;
    
}

Matrix NeuralNetwork::forward_batch(Matrix& input) const {
    Matrix output = input ;
    for (const auto& layer : layers) {
        // for(int i=0;i<output[0].size();i++)
        // {
        //     std::cout<< "cpu,index:" <<i<<"va:" <<output[0][i] <<std::endl;
        // }
        // std::cout<< "cpu,index:" <<1<<"va:" <<output[0][1] <<std::endl;
        output = layer->forward_batch(output);
        
    }
    return output;
}

void NeuralNetwork::backward(Matrix& gradInput,OptimizerAdam& optim){
    // Matrix gradOutput = gradInput;
    for (int i=layers.size()-1; 0<=i; i--){
        layers[i]->update(gradInput,optim);
        gradInput = layers[i]->backward_batch(gradInput);
        // std::cout<< output[0]<<std::endl;
        // std::cout<<"cpu grad: " <<gradInput[0][0]<<std::endl;
    }
    optim.t ++;
}

void NeuralNetwork::forward_gpu(float *inp,int batch_size){
    int sz_out;
    

    float *curr_out;
    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->outputshape * batch_size;
        // for (int j =0 ; j < layer->inputshape;j++  )
        //     std::cout<< "gpu : layer:"<< i<< "index:" << j<<"value: "<<inp[j]<<std::endl;
        // std::cout<< "gpu : layer:"<< i<< "index:" << 1<<"value: "<<inp[1]<<std::endl;
        cudaMallocManaged(&curr_out, sz_out*sizeof(float));
        layer->forward_gpu(inp, curr_out,batch_size);
        // for debug 

        inp = curr_out;
    }
    cudaMallocManaged(&curr_out, sizeof(float));
    cudaFree(curr_out);
}

void  NeuralNetwork::backward_gpu(float *gradInput,OptimizerAdam& optim,int batch_size)
{
    float *curr_grad;
    int sz_out;
    for (int i=layers.size()-1; 0<=i; i--)
    {
        sz_out = layers[i]->inputshape * batch_size;
        cudaMallocManaged(&curr_grad, sz_out*sizeof(float));
        set_zero(curr_grad,sz_out);
        // std::cout<< curr_grad[0]<<std::endl;
        layers[i]->update_gpu(gradInput,optim,batch_size);
        layers[i]->backward_gpu(gradInput,curr_grad,batch_size);
        // std::cout<< output[0]<<std::endl;
        // std::cout<< "gpu grad: "<<curr_grad[0]<<std::endl;
        gradInput = curr_grad;
    }
}


void NeuralNetwork::save(std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    for (auto layer : layers) {
        if (DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer)) {

            if(is_gpu){
                //copy weights to cpu 
                for(int i=0;i<dense_layer->weights.size();i++)
                {
                    int size = dense_layer->weights[i].size();
                    cudaMemcpy(dense_layer->weights[i].data(), &(dense_layer->weights_gpu[i*size]),  size * sizeof(float), cudaMemcpyDeviceToHost);
                }
                cudaMemcpy(dense_layer->bias.data(), dense_layer->bias_gpu, dense_layer->bias.size() * sizeof(float), cudaMemcpyDeviceToHost);
            }
            // Save weights
            for (size_t i = 0; i < dense_layer->weights.size(); i++) {
                for (size_t j = 0; j < dense_layer->weights[i].size(); j++) {
                    file.write(reinterpret_cast<const char*>(&dense_layer->weights[i][j]), sizeof(float));
                }
            }

            // Save biases
            for (size_t i = 0; i < dense_layer->bias.size(); i++) {
                file.write(reinterpret_cast<const char*>(&dense_layer->bias[i]), sizeof(float));
            }
        }
    }

    file.close();

}
void NeuralNetwork::to_gpu(void)
{
    is_gpu = true;
    for (auto& layer : layers) {
            if (DenseLayer* denseLayer = dynamic_cast<DenseLayer*>(layer)) {
                // 分配 GPU 内存
                denseLayer->sz_weights = denseLayer->weights.size() * denseLayer->weights[0].size();
                cudaMallocManaged(&denseLayer->weights_gpu, denseLayer->sz_weights* sizeof(float));
                // cudaMallocManaged(&denseLayer->cp_weights, denseLayer->sz_weights* sizeof(float));
                cudaMallocManaged(&denseLayer->bias_gpu, denseLayer->bias.size() * sizeof(float));
                float *tmp = matrix_to_float_ptr(denseLayer->weights);
                // 将数据从主机内存复制到 GPU 内存中
                cudaMemcpy(denseLayer->weights_gpu, tmp, denseLayer->sz_weights* sizeof(float), cudaMemcpyHostToDevice);
                // cudaMemcpy(denseLayer->cp_weights, tmp, denseLayer->sz_weights* sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(denseLayer->bias_gpu, denseLayer->bias.data(), denseLayer->bias.size() * sizeof(float), cudaMemcpyHostToDevice);
            }
        }
}

NeuralNetwork::~NeuralNetwork(void)
{
    if (is_gpu == true)
        for (auto& layer : layers) {
            if (DenseLayer* denseLayer = dynamic_cast<DenseLayer*>(layer)) {
                // 释放 GPU 内存
                cudaFree(denseLayer->weights_gpu);
                cudaFree(denseLayer->bias_gpu);
            }
        }
}

