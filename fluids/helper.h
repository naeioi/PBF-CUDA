#pragma once
#include <helper_math.h>

#define M_PI 3.14159265359
#define LIM_EPS 1e-2
// Should be less than delta_q^2, otherwise tensile corrective force will be inf. 
#define KERNAL_EPS 1e-4
const int MAX_PARTICLE_NUM = 120000;

typedef unsigned int uint;

__host__ __device__
inline int ceilDiv(int a, int b) { return (int)((a + b - 1) / b); }

__host__ __device__
inline float norm2(float3 u) { return u.x * u.x + u.y * u.y + u.z * u.z; }

void fexit(const int code = -1, const char* msg = nullptr);

void checkGLErr();

// #define DEBUG_NO_HASH_GRID