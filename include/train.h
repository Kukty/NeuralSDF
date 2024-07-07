#ifndef TRAIN_H
#define TRAIN_H

#include "neural_network.h"
#include "utils.h"

float get_cosine_lr(int epoch, int epochs, float base_lr, float final_lr) {
    float lr = 0.5 * (base_lr + final_lr) + 0.5 * (base_lr - final_lr) * cos(epoch * M_PI / epochs);
    return lr;
}

void train_cpu(NeuralNetwork network,Dataloader dataloader,int epochs)
{   
    MSE loss_fun;
    auto start = std::chrono::steady_clock::now();
    OptimizerAdam optim(5e-4,0.9,0.99,1e-8);
    std::cout<<"Start Training " << epochs << " epoches"<<std::endl;
    for(int epoch = 0; epoch < epochs;epoch ++ )
    {
        
        for(int index = 0; index < dataloader.len;index++)
        {
            BatchResult data = dataloader.Get_batch(index);
            Matrix input = data.input;
            Vector target = data.target;
            Matrix predict = network.forward_batch(input);
            Vector predict_v = matrixToVector(predict);
            // Vector cpu_output = network.forward(input[0]);
            float loss_value = loss_fun.forward(predict_v,target);
            Matrix grad = loss_fun.backward(predict,target);
            network.backward(grad,optim);
            std::cout<< " loss : " <<loss_value<<" step :  "<<index <<" epoch : " << epoch << " lr : " << optim.learning_rate<< std::endl;
            
        }
        if ((epoch+1)%100 ==0)
        {
            network.save("sdf_trained_weights.bin");
            std::cout<< " model saved, epoch: " << epoch<<std::endl; 
        } 
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "CPU Train " << epochs << "epochs took " << elapsed_seconds.count() << " seconds." << std::endl;
}

void train_gpu(NeuralNetwork network,Dataloader dataloader,int epochs)
{
    MSE loss_fun;
    int sz_inp = dataloader.batch_size*3;
    float *cp_inp,*out;
    Vector out_cpu(dataloader.batch_size,0);
    float loss_value;
    cudaMallocManaged(&cp_inp, sz_inp*sizeof(float));
    OptimizerAdam optim(1e-4,0.9,0.99,1e-8);
    optim.to_gpu();
    float *target_gpu;
    auto start = std::chrono::steady_clock::now();
    cudaMallocManaged(&target_gpu,dataloader.batch_size * sizeof(float));
    //loss_barckward
    float *gradOutput;
    cudaMallocManaged(&gradOutput, dataloader.batch_size*sizeof(float));
    for(int epoch = 0; epoch < epochs;epoch ++ )
    {
        
        for(int index = 0; index < dataloader.len;index++)
        {
            BatchResult data = dataloader.Get_batch(index);
            Matrix input = data.input;
            // for(int i=0;i<input.size();i++)
            // {
            //     for(int j =0;j<input[0].size();j++)
            //     {
            //         input[i][j] = 1;
            //     }
            // }
            Vector target = data.target;
            
            for(int i=0;i<dataloader.batch_size;i++)
            {
                target_gpu[i] = target[i];
                for( int j =0;j < 3;j++)
                {
                    cp_inp[i*3 + j] = input[i][j];
                }
            }
            // float *inp = matrix_to_float_ptr(input);
            // set_eq(cp_inp, inp, sz_inp);
            
            network.forward_gpu(cp_inp,dataloader.batch_size);
            out = network.layers.back()->out;

            //copy output to cpu 
            cudaMemcpy(out_cpu.data(), out, dataloader.batch_size* sizeof(float), cudaMemcpyDeviceToHost);
            loss_value = loss_fun.forward(out_cpu,target);
            // loss_value = 0.01;
            // ------------------------------------------

            
            set_zero(gradOutput,dataloader.batch_size);
            loss_fun.backward_gpu(out,target_gpu,gradOutput,dataloader.batch_size);
            network.backward_gpu(gradOutput,optim,dataloader.batch_size);

            //----------------------------------test on cpu --------------------------------------------------------------
            
            // Matrix predict = network.forward_batch(input);
            // // std::cout << "gpu output" << out[0] <<"cpu output" << predict[0][0] << std::endl;

            // Vector predict_v = matrixToVector(predict);
            // // Vector cpu_output = network.forward(input[0]);
            // float loss_value_cpu = loss_fun.forward(predict_v,target);
            // Matrix grad = loss_fun.backward(predict,target);
            // network.backward(grad,optim);             
            // std::cout<<"loss cpu: " << loss_value_cpu << " loss gpu: " <<loss_value<<" step :  "<<index <<" epoch : " << epoch << " lr : " << optim.learning_rate<< std::endl;

            //----------------------------------test on cpu --------------------------------------------------------------
            
            std::cout <<"GPU TRAIN: LOSS: " << loss_value<<" step :  "<<index <<" epoch : " << epoch << " lr : " << optim.learning_rate<< std::endl;
        }
        if ((epoch+1)%50 ==0)
        {
            network.save("sdf_trained_weights_gpu.bin");
            std::cout<< " model saved, epoch: " << epoch<<std::endl; 
        } 
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "GPU Train " << epochs << " epochs took " << elapsed_seconds.count() << " seconds." << std::endl;
}

#endif