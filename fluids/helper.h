#pragma once
#include <helper_math.h>

#define M_PI 3.14159265359
#define LIM_EPS 1e-2
#define KERNAL_EPS 1e-3
const int MAX_PARTICLE_NUM = 1 << 20;

typedef unsigned int uint;

__host__ __device__
inline int ceilDiv(int a, int b) { return (int)((a + b - 1) / b); }

void fexit(const int code = -1, const char* msg = nullptr);

void checkGLErr();