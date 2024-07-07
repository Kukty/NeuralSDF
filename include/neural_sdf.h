#ifndef NEURAL_SDF_H
#define NEURAL_SDF_H
#include "neural_network.h"
#include "utils.h"
#include "module.h"
#include "public_camera.h"
#include "public_scene.h"
#include <cmath>
#include <chrono>
#include <omp.h>

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
const size_t IMAGE_WIDTH = 256;
const size_t IMAGE_HEIGHT = 256;
const size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
float aspectRatio = (float) IMAGE_WIDTH / IMAGE_HEIGHT;


std::vector<float> RayTrace_cpu(const SdfScene& scene, const Camera& cam, size_t threadCount = 1) {
    float fovTanHalf = std::tan(cam.fov_rad * 0.5f);
    std::vector<float> cpu_image(IMAGE_SIZE, 0.0f);

    omp_set_num_threads(threadCount);
    auto startTime = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t widthIndex = 0; widthIndex < IMAGE_WIDTH; ++widthIndex) {
        for (size_t heightIndex = 0; heightIndex < IMAGE_HEIGHT; ++heightIndex) {
            float normX = (2.0f * widthIndex + 1) / IMAGE_WIDTH - 1;
            float normY = (2.0f * heightIndex + 1) / IMAGE_HEIGHT - 1;
            float3_cpu direction = (cam.lookDirection + cam.right * normX * fovTanHalf * aspectRatio + cam.up * normY * fovTanHalf).normalize();
            float init_t = 0 ;
            bool flag = rayIntersectsBox(cam.pos,direction,float3_cpu{-1,-1,-1},float3_cpu{1,1,1},init_t);
            if (flag == false){
                continue;
            }

            float3_cpu v = cam.pos+  direction * init_t;
            float3_cpu col;

            float3_cpu intersection = {0,0,0};
            float3_cpu pixelColor;
            
            do{
                float distance = scene.DistanceEvaluation(v);
                if (std::abs(distance) < 0.0001)
                {
                    col = {1.0,1.0,1.0};
                    intersection = v;
                    break;
                }
                v = v + direction * distance;
                // std::cout<<v.length()<<std::endl;
            }while (v.length()<1.1);

            if (intersection.length() > 0) 
            {
                    float3_cpu surfaceNormal = scene.EstimateNormal(intersection);
                    if (scene.light) 
                    {
                        col = col * std::max(0.1f, surfaceNormal.dot(scene.light->dir));
                    }
                    pixelColor = col;
                    // std::cout << pixelColor.x <<std::endl;
                size_t idx = 3 * (heightIndex * IMAGE_WIDTH + widthIndex);
                cpu_image[idx] = pixelColor.x;
                cpu_image[idx + 1] = pixelColor.y;
                cpu_image[idx + 2] = pixelColor.z;
            }

  
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endTime - startTime;
    std::cout << "Время выполнения алгоритма на CPU " << threadCount << " поток: " << duration.count() / 1000 << "s" << std::endl;

    return cpu_image;
}



// std::vector<float> RayTrace_GPU(const SdfScene& scene, const Camera& cam) {
//     float fovTanHalf = tan(cam.fov_rad * 0.5f);
//     float* cpu_image;
//     cudaMallocManaged(&cpu_image, IMAGE_SIZE * sizeof(float));
//     auto startTime = std::chrono::high_resolution_clock::now();
//     float3 lookDirection_cam = make_float3(cam.lookDirection.x,cam.lookDirection.y,cam.lookDirection.z);
//     float3 right_cam = make_float3(cam.right.x,cam.right.y,cam.right.z);
//     float3 up_cam = make_float3(cam.up.x,cam.up.y,cam.up.z);
//     float3 pos_cam = make_float3(cam.pos.x,cam.pos.y,cam.pos.z);
//     float3 light_dir = make_float3(scene.light->dir.x,scene.light->dir.y,scene.light->dir.z);
//     run_gpu_raytrace(cpu_image,lookDirection_cam,right_cam,up_cam,pos_cam,IMAGE_WIDTH,IMAGE_HEIGHT,fovTanHalf,aspectRatio,light_dir,scene.network);

//     std::vector<float> renderedImage_vect(IMAGE_SIZE,0.0);
//     for (size_t i = 0; i < IMAGE_SIZE; ++i) {
//         renderedImage_vect[i] = cpu_image[i];
//     }
//     return renderedImage_vect;
// }

#endif