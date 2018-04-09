#include "helper.h"

__global__ void advect_kernel(
	float3 *pos, float3 *npos, float3 *vel, float3 *nvel, 
	int nparticle, float dt, float3 g) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i < nparticle) {
		nvel[i] = vel[i] + dt * g;
		npos[i] = pos[i] + dt * nvel[i];
	}
}

__global__ void computeGridRange(uint* gridIds, uint* gridStart, uint* gridEnd, int n) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= n) return;

	extern __shared__ uint pre[];

	pre[threadIdx.x + 1] = gridIds[i];
	//pre[0] = cellIds[i];
	if (threadIdx.x == 0)
		pre[0] = i == 0 ? (uint)-1 : gridIds[i - 1];
	__syncthreads();

	uint current = pre[threadIdx.x+1], last = pre[threadIdx.x];
	// uint current = 0, last = 0;
	
	if (i == n - 1) {
		gridEnd[current] = i;
		return;
	}

	if (current != last) {
		gridStart[current] = i;
		if(last != (uint)-1)
			gridEnd[last] = i;
	}
}

__device__
float h_poly6(float h, float r2) {
	float h2 = h * h;
	float h3 = h2 * h;
	float h9 = h3 * h3;
	float coef = 315.f / (64.f * M_PI *  h9);
	return coef * (h2 - r2) * (h2 - r2) * (h2 - r2);
}

__device__ 
float3 h_spikyGrad(float h, float3 r) {
	float h6 = h * h;
	h6 = h6 * h6 * h6;
	float coef = -45.f / (M_PI * h6);
	float rlen = length(r);
	return (coef * (h - rlen) * (h - rlen)) * normalize(r);
}


template </* typename Func1, typename Func2, */ typename Func3, typename Func4>
__global__
void computeLambda(
	float* lambdas, float3* grads,
	uint* cellIds, uint* cellStarts, uint* cellEnds,
	uint3 cellDim,
	float3* pos, uint n, float pho0, float lambda_eps,
	/* Func1 poly6, Func2 spikyGrad, */ float h,
	Func3 posToCellxyz, Func4 cellxyzToId,
	uint* maxStart, uint* maxEnd) {
	
	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (i >= n) return;

	/* -- Compute lambda -- */

	int3 ind = posToCellxyz(pos[i]);
	float pho = 0.f;
	float3 grad = make_float3(0, 0, 0);
// #pragma unroll 3
	for (int dx = -1; dx <= 1; dx++) {
// #pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
// #pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int x = ind.x + dx, y = ind.y + dy, z = ind.z + dz;
				int cellId = cellxyzToId(x, y, z);
				uint start = cellStarts[cellId], end = cellEnds[cellId];
				// atomicMax(maxStart, start);
				// atomicMax(maxEnd, end); 
				for (int j = start; j < end; j++) if (j != i) {
					float3 d = pos[i] - pos[j];
					float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
					pho += h_poly6(h, r2);
					/* TODO: call to spikyGrad will crash the kernel */
					grad += h_spikyGrad(h, d);
				}
			}
		}
	}

	/* TODO: The formula here is different from that on the paper. check which works better */
	lambdas[i] = -(pho - pho0) / (grad.x * grad.x + grad.y * grad.y + grad.z * grad.z + lambda_eps);
	grads[i] = grad;
}