#include "helper.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void advect_kernel(
	float3 *pos, float3 *npos, float3 *vel, float3 *nvel, 
	int nparticle, float dt, float3 g,
	float3 ulim, float3 llim) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i < nparticle) {
		float3 npos_;
		nvel[i] = vel[i] + dt * g;
		npos_ = pos[i] + dt * nvel[i];
		npos_.x = fmaxf(min(npos_.x, ulim.x), llim.x);
		npos_.y = fmaxf(min(npos_.y, ulim.y), llim.y);
		npos_.z = fmaxf(min(npos_.z, ulim.z), llim.z);
		npos[i] = npos_;
		// printf("(%f,%f,%f) => (%f,%f,%f)\n", pos[i].x, pos[i].y, pos[i].z, npos[i].x, npos[i].y, npos[i].z);
	}
}

__global__ void computeGridRange(uint* gridIds, uint* gridStart, uint* gridEnd, int n) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= n) return;

	extern __shared__ uint pre[];

	pre[threadIdx.x + 1] = gridIds[i];
	// printf("gridIds[(%d, %d)=%d] = %u\n", blockIdx.x, threadIdx.x, i, gridIds[i]);

	pre[0] = i == 0 ? (uint)-1 : gridIds[i - 1];
	__syncthreads();

	uint current = pre[threadIdx.x+1], last = pre[threadIdx.x];
	// uint current = 0, last = 0;
	
	if (i == n - 1) {
		gridEnd[current] = i + 1;
		// printf("gridEnd[%u]=%d\n", current, i);
		return;
	}

	if (current != last) {
		gridStart[current] = i;
		// printf("gridStart[%u]=%d\n", current, i);
		if (last != (uint)-1) {
			gridEnd[last] = i;
			// printf("gridEnd[%u]=%d\n", last, i);
		}
	}
}

__device__
float h_poly6(float h, float r2) {
	float h2 = h * h;
	if (r2 >= h2 || r2 < KERNAL_EPS) return 0;
	float h3 = h2 * h;
	float h9 = h3 * h3;
	float coef = 315.f / (64.f * M_PI *  h9);
	return coef * (h2 - r2) * (h2 - r2) * (h2 - r2);
}

__device__ 
float3 h_spikyGrad(float h, float3 r) {
	float rlen = length(r);
	if (rlen >= h || rlen < KERNAL_EPS) return make_float3(0.f, 0.f, 0.f);
	float h6 = h * h;
	h6 = h6 * h6 * h6;
	float coef = -45.f / (M_PI * h6);
	return (coef * (h - rlen) * (h - rlen)) * normalize(r);
}


template </* typename Func1, typename Func2, */ typename Func3, typename Func4>
__global__
void computeLambda(
	float* lambdas, /*float* grads_l2,*/
	uint* cellIds, uint* cellStarts, uint* cellEnds,
	uint3 cellDim,
	float3* pos, uint n, float pho0, float lambda_eps,
	/* Func1 poly6, Func2 spikyGrad, */ float h,
	Func3 posToCellxyz, Func4 cellxyzToId) {
	
	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (i >= n) return;
	if (i == 0) printf("(%f, %f, %f)\n", pos[i].x, pos[i].y, pos[i].z);
	/* -- Compute lambda -- */

	int3 ind = posToCellxyz(pos[i]);
	float pho = 0.f, gradj_l2 = 0.f, grad_l2;
	float3 gradi = make_float3(0, 0, 0), grad;
// #pragma unroll 3
	for (int dx = -1; dx <= 1; dx++) {
// #pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
// #pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int x = ind.x + dx, y = ind.y + dy, z = ind.z + dz;
				int cellId = cellxyzToId(x, y, z);
				uint start = cellStarts[cellId], end = cellEnds[cellId];
				for (int j = start; j < end; j++) if (j != i) {
					float3 d = pos[i] - pos[j];
					float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
					pho += h_poly6(h, r2);
					/* TODO: call to spikyGrad will crash the kernel */
					grad = h_spikyGrad(h, d) / pho0;
					gradi += grad;
					gradj_l2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
				}
			}
		}
	}

	grad_l2 = gradj_l2 + gradi.x * gradi.x + gradi.y * gradi.y + gradi.z * gradi.z;	
	lambdas[i] = -(pho / pho0 - 1) / (grad_l2 + lambda_eps);
}

template <typename Func1, typename Func2>
__global__
void computedpos(
	float* lambdas, /*float* grads_l2,*/
	uint* cellIds, uint* cellStarts, uint* cellEnds,
	uint3 cellDim,
	float3* pos, float3* dpos, uint n, float pho0,
	/* Func1 poly6, Func2 spikyGrad, */ float h,
	Func1 posToCellxyz, Func2 cellxyzToId) {

	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i >= n) return;
	// if (i == 0) printf("(%f,%f,%f)\n", pos[i].x, pos[i].y, pos[i].z);

	// #pragma unroll 3
	int3 ind = posToCellxyz(pos[i]);
	float lambda = lambdas[i];
	float3 d = make_float3(0.f, 0.f, 0.f);
	for (int dx = -1; dx <= 1; dx++) {
		// #pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
			// #pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int x = ind.x + dx, y = ind.y + dy, z = ind.z + dz;
				int cellId = cellxyzToId(x, y, z);
				uint start = cellStarts[cellId], end = cellEnds[cellId];
				for (int j = start; j < end; j++) if (j != i) {
					float3 p = pos[i] - pos[j];
					d += (lambda + lambdas[j]) * h_spikyGrad(h, p);
					if (0 && i == 0) {
						float3 sgrad = h_spikyGrad(h, p);
						if (sgrad.x != 0.f)
							printf("sgrad(%f,%f,%f)=(%f,%f,%f)\n", p.x, p.y, p.z, sgrad.x, sgrad.y, sgrad.z);
					}
				}
			}
		}
	}

	dpos[i] = d / pho0;
}