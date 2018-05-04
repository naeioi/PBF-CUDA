#pragma once
#include <helper_math.h>

#define M_PI 3.14159265359
#define LIM_EPS 1e-3
// Should be less than delta_q^2, otherwise tensile corrective force will be inf. 
#define KERNAL_EPS 1e-4
// Maximum delta_p in correntDensity()
#define MAX_DP 0.1
const int MAX_PARTICLE_NUM = 120000;

typedef unsigned int uint;

__host__ __device__
inline int ceilDiv(int a, int b) { return (int)((a + b - 1) / b); }

__host__ __device__
inline float norm2(float3 u) { return u.x * u.x + u.y * u.y + u.z * u.z; }

__host__ __device__
inline float3 clamp3f(float3 u, float3 llim, float3 ulim) {
	return make_float3(fmaxf(fminf(u.x, ulim.x), llim.x), fmaxf(fminf(u.y, ulim.y), llim.y), fmaxf(fminf(u.z, ulim.z), llim.z));
}

__host__ __device__
inline float3 clamp3f(float3 u, float llim, float ulim) {
	return clamp3f(u, make_float3(llim, llim, llim), make_float3(ulim, ulim, ulim));
}

void fexit(const int code = -1, const char* msg = nullptr);

void checkGLErr();

#define expand(p) p.x, p.y, p.z

// #define DEBUG_NO_HASH_GRID