// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include "module.h"


class NeuralNetwork {
public:
    std::vector<Module*> layers;
    bool is_gpu = false;

public:
    NeuralNetwork(const std::vector<std::string>& architecture);
    NeuralNetwork(const std::vector<std::string>& architecture,const std::string &weightsfilename);
    Vector forward(Vector& input) const;
    Matrix forward_batch(Matrix& input) const ;
    void forward_gpu(float *inp,int batch_size);
    void backward_gpu(float *gradInput,OptimizerAdam& optim,int batch_size);
    void backward(Matrix& gradInput,OptimizerAdam& optim);
    void save(std::string filename);
    void to_gpu(void);
    ~NeuralNetwork(void);
};

void run_gpu_raytrace(float* image,float3 lookDirection,float3 right,float3 up,float3 pos, int image_width,int image_height,float fovTanHalf,float aspectRatio,float3 light_dir,NeuralNetwork network_gpu);

#endif // NEURAL_NETWORK_H