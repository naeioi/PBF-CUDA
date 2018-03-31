#include "helper.h"

__global__ void advect_kernel(
	float3 *pos, float3 *npos, float3 *vel, float3 *nvel, 
	int nparticle, float dt, float g) {
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
	if (threadIdx.x == 0)
		pre[0] = i == 0 ? (uint)-1 : gridIds[i - 1];
	__syncthreads();

	uint current = pre[threadIdx.x+1], last = pre[threadIdx.x];
	
	if (i == n - 1) {
		gridEnd[current] = i;
		return;
	}

	if (current != last) {
		gridStart[current] = gridEnd[last] = i;
	}
}

template <typename Func1, typename Func2, typename Func3, typename Func4>
__global__
void computeLambda(
	float* lambdas, float3* grads,
	uint* gridIds, uint* gridStarts, uint* gridEnd,
	uint3 gridDim,
	float3* pos, uint n, float pho0, float lambda_eps,
	Func1 poly6, Func2 spikyGrad,
	Func3 getGridxyz, Func4 xyzToId) {
	
	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (i >= n) return;

	/* -- Compute lambda -- */

	uint3 ind = getGridxyz(pos[i]);
	float pho = 0.f;
	float3 grad = make_float3(0, 0, 0);
#pragma unroll 3
	for (int dx = -1; dx <= 1; dx++) {
		int x = ind.x + dx;
		if (x < 0 || x >= gridDim.x) continue;
#pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
			int y = ind.y + dy;
			if (y < 0 || y >= gridDim.y) continue;
#pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int z = ind.z + dz;
				if (z < 0 || z >= gridDim.z) continue;
				uint gridId = xyzToId(x, y, z);
				uint start = gridStarts[gridId], end = gridEnd[gridId];
				for (int j = start; j < end; j++) if (j != i) {
					float3 d = pos[i] - pos[j];
					float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
					pho += poly6(r2);
					grad += spikyGrad(d);
				}
			}
		}
	}

	/* TODO: The formula here is different from that on the paper. check which works better */
	lambdas[i] = -(pho - pho0) / (grad.x * grad.x + grad.y * grad.y + grad.z * grad.z + lambda_eps);
	grads[i] = grad;
}