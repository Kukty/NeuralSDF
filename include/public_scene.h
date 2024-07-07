#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "neural_network.h"
#include <cerrno>



class DirectedLight {
public:
    float3_cpu dir{}; //direction TO light, i.e 0,1,0 if light is above
    float intensity = 1.0f;

    bool toFile(const char *filename)
    {
        FILE *f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr, "failed to open/create file %s. Errno %d\n", filename, (int) errno);
            return false;
        }
        fprintf(f, "light direction = %f, %f, %f\n", dir.x, dir.y, dir.z);
        fprintf(f, "intensity = %f\n", intensity);

        int res = fclose(f);
        if (res != 0) {
            fprintf(stderr, "failed to close file %s. fclose returned %d\n", filename, res);
            return false;
        }
        return true;
    }

    bool fromFile(const char *filename) {
        FILE *f = fopen(filename, "radius");
        if (!f) {
            fprintf(stderr, "failed to open file %s. Errno %d\n", filename, (int) errno);
            return false;
        }
        fscanf(f, "light direction = %f, %f, %f\n", &dir.x, &dir.y, &dir.z);
        fscanf(f, "intensity = %f\n", &intensity);

        int res = fclose(f);
        if (res != 0) {
            fprintf(stderr, "failed to close file %s. fclose returned %d\n", filename, res);
            return false;
        }
        return true;
    }
};

class SdfScene {

public:
    NeuralNetwork network;
    DirectedLight* light;

    SdfScene(const NeuralNetwork& network):network(network){};

    bool loadLight(const char *filename) {
        light = new DirectedLight();
        return light->fromFile(filename);
    }

    float DistanceEvaluation(const float3_cpu &p) const {
        Vector p_input = {p.x,p.y,p.z};
        float dis = network.forward(p_input)[0];
        return dis;
    }

    // float DistanceEvaluation_gpu(const float3_cpu &p) const{
    //     float *input;
    //     Vector point = {p.x,p.y,p.z};
    //     cudaMalloc((void**)&input, 3 * sizeof(float));
    //     cudaMemcpy(input, point.data(), 3* sizeof(float), cudaMemcpyHostToDevice);
    //     float* output_gpu = network.forward_gpu(input);
    //     float *output_cpu = new float[1];
    //     cudaMemcpy(output_cpu, output_gpu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaFree(input);
    //     cudaFree(output_gpu);
    //     return output_cpu[0];
    // }


    float3_cpu EstimateNormal(const float3_cpu& z, float eps = 0.0001) const {
        float3_cpu z1 = z + float3_cpu(eps, 0, 0);
        float3_cpu z2 = z - float3_cpu(eps, 0, 0);
        float3_cpu z3 = z + float3_cpu(0, eps, 0);
        float3_cpu z4 = z - float3_cpu(0, eps, 0);
        float3_cpu z5 = z + float3_cpu(0, 0, eps);
        float3_cpu z6 = z - float3_cpu(0, 0, eps);
        float dx = DistanceEvaluation(z1) - DistanceEvaluation(z2);
        float dy = DistanceEvaluation(z3) - DistanceEvaluation(z4);
        float dz = DistanceEvaluation(z5) - DistanceEvaluation(z6);
        return (float3_cpu(dx, dy, dz) / (2.0f * eps)).normalize();
    }

};