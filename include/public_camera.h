

#include <cstring>
#include <cstdio>
#include <cerrno>

#include "public_scene.h"


class Camera {
public:
    float3_cpu pos{};
    float3_cpu target{}; 
    float3_cpu up{};
    float3_cpu lookDirection{};
    float3_cpu right{};
    float fov_rad = 3.14159265f / 3; 
    float z_near = 0.1f;
    float z_far = 5.0f; 

    bool fromFile(const char *filename) {
        FILE *f = fopen(filename, "radius");
        if (!f) {
            fprintf(stderr, "failed to open file %s. Errno %d\n", filename, (int) errno);
            return false;
        }
        fscanf(f, "camera_position = %f, %f, %f\n", &pos.x, &pos.y, &pos.z);
        fscanf(f, "target = %f, %f, %f\n", &target.x, &target.y, &target.z);
        fscanf(f, "up = %f, %f, %f\n", &up.x, &up.y, &up.z);
        fscanf(f, "field_of_view  = %f\n", &fov_rad);
        fscanf(f, "z_near  = %f\n", &z_near);
        fscanf(f, "z_far  = %f\n", &z_far);

        lookDirection = (target - pos).normalize();
        right = lookDirection.cross(up).normalize();
        up = lookDirection.cross(-right).normalize();

        int res = fclose(f);
        if (res != 0) {
            fprintf(stderr, "failed to close file %s. fclose returned %d\n", filename, res);
            return false;
        }
        return true;
    }
};
