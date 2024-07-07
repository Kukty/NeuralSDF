#include <cmath>
#include <iostream>
#include "utils.h"

struct float3_cpu {
    float x, y, z;
    
    float3_cpu() : x(0), y(0), z(0) {}

    float3_cpu(float x, float y, float z) : x(x), y(y), z(z) {
    }

    float3_cpu operator+(const float3_cpu& v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    float3_cpu operator-(const float3_cpu& v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    float3_cpu operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    
    float3_cpu operator-(float s) const {
        return {x - s, y - s, z - s};
    }

    float3_cpu operator/(float s) const {
        return {x / s, y / s, z / s};
    }

    float3_cpu operator-() const {
        return {-x, -y, -z};
    }

    float3_cpu abs() const {
        return {std::abs(x), std::abs(y), std::abs(z)};
    }

    float3_cpu max(float upperBound) const {
        return {std::max(x, upperBound), std::max(y, upperBound), std::max(z, upperBound)};
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    float dot(const float3_cpu& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    float3_cpu normalize() const {
        float length = this->length();
        float invLen = 1.0f;
        if (length != 0) {
            invLen /= length;
        }
        return {x * invLen, y * invLen, z * invLen};
    }

    float3_cpu cross(const float3_cpu& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }

    float& operator[](int index) {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else if (index == 2) {
            return z;
        } else {
            throw std::out_of_range("Invalid index. Index must be 0, 1, or 2.");
        }
    }

    const float& operator[](int index) const {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else if (index == 2) {
            return z;
        } else {
            throw std::out_of_range("Invalid index. Index must be 0, 1, or 2.");
        }
    }
};
