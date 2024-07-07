#pragma once

#include <string>
#include <vector>

void read_image_rgb(std::string path, std::vector<float> &image_data, int &width, int &height);
void write_image_rgb(std::string path, const std::vector<float> &image_data, int width, int height);