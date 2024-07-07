#include <chrono>
#include <iostream>
#include <random>
#include "include/train.h"
#include "include/utils.h"
#include "include/module.h"
#include "include/neural_network.h"
#include "include/neural_sdf.h"
#include "include/stb_image.h"
#include "include/public_image.h"
#include "include/neural_sdf.h"


int main(int argc, char *argv[]){

    std::cout<<"---------------------------------------------Шаг 1. SIREN (10 баллов) ---------------------------------------------------------------" << std::endl;
    Matrix points;
    Vector referenceDistances;
    std::string filename = argv[1]; 
    if (!readSdfTestFile(filename, points, referenceDistances)) {
        return 1; // Exit if file reading failed
    }
    std::vector<std::string> architecture;
    std::string archfilename = argv[2]; 
    if (!readSdfArchFile(archfilename, architecture)) {
        return 1; // Exit if file reading failed
    }
    std::string weightsfilename = argv[3]; 
    NeuralNetwork network(architecture,weightsfilename);
    
    auto start = std::chrono::steady_clock::now();
    float err = 0.0;
    for (int i=0;i<50;i++) {

            Vector cpu_output = network.forward(points[i]);
            err += std::abs(referenceDistances[i] - cpu_output[0]);
            // std::cout<<std::abs(referenceDistances[i] - cpu_output[0])<<std::endl;
        }
    err = err / points.size();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Network in cpu forward of  " << points.size() << " points took "<< elapsed_seconds.count() << " seconds. Mean error with reference: " <<err<< std::endl;
    // /// ------------------------------Test GPU -----------------------------------------------------///
    // network.to_gpu();
    // // input to gpu
    // start = std::chrono::steady_clock::now();
    // float *input;
    // cudaMalloc((void**)&input, 3 * sizeof(float));
    // float *output_cpu = new float[1];
    // for (const auto& point : points) {
        
    //     cudaMemcpy(input, point.data(), 3* sizeof(float), cudaMemcpyHostToDevice);
    //     float* output_gpu = network.forward_gpu(input);
    //     // transfer output to cpu 
        
    //     cudaMemcpy(output_cpu, output_gpu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    //     // std::cout<< "GPU out put: " << output_cpu[0] <<std::endl;
    //     cudaFree(output_gpu);
    // }
    // cudaFree(input);
    
    // end = std::chrono::steady_clock::now();
    // elapsed_seconds = end - start;
    // std::cout << "Network in gpu forward of  " << points.size() << " points took "<< elapsed_seconds.count() << " seconds." << std::endl;
    /// ------------------------------Test GPU -----------------------------------------------------///

    ///------------------------------------------------------ ------------------------------------------------
    std::cout<<"---------------------------------------------Шаг 2. Визуализация (5 баллов) ---------------------------------------------------------------" << std::endl;
    Camera camera = Camera();
    SdfScene scene = SdfScene(network);

    camera.fromFile(argv[4]);
    scene.loadLight(argv[5]);
    std::vector<float> cpuOutImage = RayTrace_cpu(scene, camera, 16);

    std::string cpu_filename = "cpu_sdf1_mytrain.png";
    write_image_rgb(cpu_filename, cpuOutImage, IMAGE_WIDTH, IMAGE_HEIGHT);
    std::cout<<"---------------------------------------------Шаг 3. Обучение (10 баллов) ---------------------------------------------------------------" << std::endl;
    Matrix train_points;
    Vector train_referenceDistances;
    std::string train_filename = argv[6]; 
    if (!readSdfTestFile(train_filename, train_points, train_referenceDistances)) {
        return 1; // Exit if file reading failed
    }
    Dataloader train_dataloader(train_points,train_referenceDistances,512,false);
    MSE loss_fun;
    int epoches = 151 ;
    NeuralNetwork network_init(architecture);
    
    // NeuralNetwork network_init(architecture,weightsfilename);
    train_cpu(network_init,train_dataloader,epoches);
    std::cout<<"---------------------------------------------Шаг 4. Перенос прямого прохода на GPU (5 баллов) ---------------------------------------------------------------" << std::endl;
    
    // std::vector<float> gpuOutImage = RayTrace_GPU(scene, camera);

    // std::string gpu_filename = "gpu_sdf1_mytrain.png";
    // write_image_rgb(gpu_filename, gpuOutImage, IMAGE_WIDTH, IMAGE_HEIGHT);
    std::cout<<"---------------------------------------------Шаг 5. Перенос обратного прохода на GPU (10 баллов) ---------------------------------------------------------------" << std::endl;
    
    network_init.to_gpu();
    train_gpu(network_init,train_dataloader,epoches);

    // network.to_gpu();
    // std::vector<float> gpuOutImage = RayTrace_gpu(scene, camera, 1);
    // std::string gpu_filename = "gpu_out.png";
    // write_image_rgb(gpu_filename, gpuOutImage, IMAGE_WIDTH, IMAGE_HEIGHT);
    return 0;
}
