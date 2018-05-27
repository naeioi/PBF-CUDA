#include "helper.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MI 966

__global__ void advect_kernel(
	uint* iids,
	float3 *pos, float3 *npos, float3 *vel,
	int nparticle, float dt, float3 g,
	float3 ulim, float3 llim) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i < nparticle) {
		vel[i] += dt * g;
		npos[i] = pos[i] + dt * vel[i];	
		if (0 && iids[i] == MI)
			printf("#%-3d vel=(%.3f,%.3f,%.3f), pos=(%.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f)\n", iids[i], expand(vel[i]), expand(pos[i]), expand(npos[i]));
	}
}

__global__ void computeGridRange(uint* gridIds, uint* gridStart, uint* gridEnd, int n) {
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	extern __shared__ uint pre[];

	if (i < n) {
		pre[threadIdx.x + 1] = gridIds[i];
		if (threadIdx.x == 0)
			pre[0] = i == 0 ? (uint)-1 : gridIds[i - 1];
	}

	__syncthreads();

	if (i < n) {
		uint current = pre[threadIdx.x + 1], last = pre[threadIdx.x];

		if (current != last) {
			gridStart[current] = i;
			// printf("gridStart[%d]=%d\n", current, i);
			if (last != (uint)-1) {
				gridEnd[last] = i;
				// printf("gridEnd[%d]=%d\n", last, i);
			}
		}

		if (i == n - 1) {
			gridEnd[current] = n;
			// printf("gridEnd[%d]=%d\n", current, i);
		}
	}
}

__host__ __device__
float h_poly6(float h, float r2) {
	float h2 = h * h;
	if (r2 >= h2) return 0;
	float ih9 = powf(h, -9.f);
	float coef = 315.f * ih9 / (64.f * M_PI);
	return coef * (h2 - r2) * (h2 - r2) * (h2 - r2);
}

__host__ __device__ 
float3 h_spikyGrad(float h, float3 r) {
	float rlen = length(r);
	if (rlen >= h || rlen < KERNAL_EPS) return make_float3(0.f, 0.f, 0.f);
	float ih6 = powf(h, -6.f);
	float coef = -45.f * ih6 / M_PI;
	return (coef * (h - rlen) * (h - rlen)) * normalize(r);
}

template <typename Func1, typename Func2, typename Func3>
__global__
void computeLambda(
	uint* iids,
	float* lambdas, float* phos,
	uint* cellIds, uint* cellStarts, uint* cellEnds,
	int3 cellDim,
	float3* pos, int n, float pho0, float lambda_eps, float k_boundaryDensity,
	float h,
	Func1 posToCellxyz, Func2 cellxyzToId, Func3 boundaryDensity) {
	
	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if (i >= n) return;

	/* -- Compute lambda -- */

	uint iid = iids[i];
	int3 ind = posToCellxyz(pos[i]);
	float pho = 0.f, gradj_l2 = 0.f, grad_l2;
	float3 gradi = make_float3(0, 0, 0), grad, cpos = pos[i];


#ifndef DEBUG_NO_HASH_GRID

#pragma unroll 3
	for (int dx = -1; dx <= 1; dx++) {
#pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
#pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int x = ind.x + dx, y = ind.y + dy, z = ind.z + dz;
				if (x < 0 || x >= cellDim.x || y < 0 || y >= cellDim.y || z < 0 || z >= cellDim.z) continue;
				int cellId = cellxyzToId(x, y, z);
				uint start = cellStarts[cellId], end = cellEnds[cellId];
				for (int j = start; j < end; j++) {
					float3 d = cpos - pos[j];
					float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
					pho += h_poly6(h, r2);
					grad = h_spikyGrad(h, d) / pho0;
					gradi += grad;
					if (0 && iid == MI) printf("#%-3d <~ #%-3d:(%f,%f,%f), pho=%.1f\n", iid, iids[j], expand(pos[j]), pho);
					/* There is always a j equals to i in theory, but because hash grid is built once for multiple rounds of position correction, 
					 * This is not guarenteed, and this leads to a wrong pho.
					 */
					if (j != i) {
						gradj_l2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
					}
				}
			}
		}
	}

#else

	for (int j = 0; j < n; j++) if (j != i) {
		float3 d = pos[i] - pos[j];
		float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
		pho += h_poly6(h, r2);
		/* TODO: call to spikyGrad will crash the kernel */
		grad = h_spikyGrad(h, d) / pho0;
		gradi += grad;
		gradj_l2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
	}
#endif

	float boundPho = k_boundaryDensity * boundaryDensity(cpos);
	pho += boundPho;

	grad_l2 = gradj_l2 + gradi.x * gradi.x + gradi.y * gradi.y + gradi.z * gradi.z;	
	lambdas[i] = -(pho / pho0 - 1) / (grad_l2 + lambda_eps);
	phos[i] = pho;

	if (0 && iid == MI)
		printf("#%-3d gradi=(%.3f,%.3f,%.3f), gradj_l2=%.3f, boundPho=%.3f, lambda=%.3f, pho=%.3f\n"
			, iids[i], expand(gradi), gradj_l2, boundPho, lambdas[i], pho);
}

template <typename Func1, typename Func2>
__global__
void computetpos(
	float* lambdas, 
	uint* iids,
	uint* cellIds, uint* cellStarts, uint* cellEnds,
	int3 cellDim,
	float3* pos, float3 *tpos, int n, 
	float pho0, float h, float coef_corr, float n_corr, 
	Func1 posToCellxyz, Func2 cellxyzToId, float3 ulim, float3 llim) {

	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i >= n) return;
	// if (i == 0) printf("(%f,%f,%f)\n", pos[i].x, pos[i].y, pos[i].z);

	uint iid = iids[i];
	int3 ind = posToCellxyz(pos[i]);
	float lambda = lambdas[i];
	float3 d = make_float3(0.f, 0.f, 0.f), cpos = pos[i];
#ifndef DEBUG_NO_HASH_GRID
#pragma unroll 3
	for (int dx = -1; dx <= 1; dx++) {
#pragma unroll 3
		for (int dy = -1; dy <= 1; dy++) {
#pragma unroll 3
			for (int dz = -1; dz <= 1; dz++) {
				int x = ind.x + dx, y = ind.y + dy, z = ind.z + dz;
				if (x < 0 || x >= cellDim.x || y < 0 || y >= cellDim.y || z < 0 || z >= cellDim.z) continue;
				int cellId = cellxyzToId(x, y, z);
				uint start = cellStarts[cellId], end = cellEnds[cellId];
				for (int j = start; j < end; j++) if (j != i) {
					float3 p = cpos - pos[j], dd;
					float corr = coef_corr * powf(h_poly6(h, norm2(p)), n_corr);
					dd = (lambda + lambdas[j] + corr) * h_spikyGrad(h, p);
					d += dd;
				}
			}
		}
	}
#else
	for (int j = 0; j < n; j++) if (j != i) {
		float3 p = pos[i] - pos[j];
		float corr = coef_corr * powf(h_poly6(h, norm2(p)), n_corr);
		d += (lambda + lambdas[j] + corr) * h_spikyGrad(h, p);
		if (0 && i == 0) {
			float3 sgrad = h_spikyGrad(h, p);
			if (sgrad.x != 0.f)
				printf("sgrad(%f,%f,%f)=(%f,%f,%f)\n", p.x, p.y, p.z, sgrad.x, sgrad.y, sgrad.z);
		}
	}
#endif
	
	d = clamp3f(d / pho0, -MAX_DP, MAX_DP);
	if (0 && iid == MI)
		printf("#%-3d pos=(%.3f,%.3f,%.3f)+(%.3f,%.3f,%.3f)=(%.3f,%.3f,%.3f)\n", iids[i], expand(cpos), expand(d), expand(cpos+d));

	cpos += d;

	cpos.x = max(min(cpos.x, ulim.x - LIM_EPS), llim.x + LIM_EPS);
	cpos.y = max(min(cpos.y, ulim.y - LIM_EPS), llim.y + LIM_EPS);
	cpos.z = max(min(cpos.z, ulim.z - LIM_EPS), llim.z + LIM_EPS);
	tpos[i] = cpos;
}

template <typename Func1, typename Func2>
__global__
void computeXSPH(
	float* phos,
	uint* iids,
	uint* cellStarts, uint* cellEnds, int3 cellDim,
	float3* pos, float3* vel, float3* nvel, int n,
	float c_XSPH, float h,
	Func1 grid_pos2xyz, Func2 grid_xyz2id) {

	int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i >= n) return;

	float cpho = phos[i];
	float3 cpos = pos[i], cvel = vel[i], avel = make_float3(0.f, 0.f, 0.f);
	int3 xyz = grid_pos2xyz(cpos);

	for (int dx = -1; dx <= 1; dx++) {
		int x = xyz.x + dx;
		// if (x < 0 || x >= cellDim.x) continue;
		for (int dy = -1; dy <= 1; dy++) {
			int y = xyz.y + dy;
			// if (y < 0 || y >= cellDim.y) continue;
			for (int dz = -1; dz <= 1; dz++) {
				int z = xyz.z + dz;
				if (x < 0 || x >= cellDim.x || y < 0 || y >= cellDim.y || z < 0 || z >= cellDim.z) continue;

				int ncell = grid_xyz2id(x, y, z), start = cellStarts[ncell], end = cellEnds[ncell];
				for (int j = start; j < end; j++) {
					float3 dp = cpos - pos[j];
					float3 vp = vel[j] - cvel;
					avel += 2.f * vp * h_poly6(h, norm2(dp)) / (cpho + phos[j]);
					// avel += 2.f * vp * 0.f / (cpho + phos[j]);
				}
			}
		}
	}

	nvel[i] = vel[i] + c_XSPH * avel;
	if (0 && iids[i] == MI) {
		printf("#%-3d nvel=(%f,%f,%f), vel=(%f,%f,%f)\n", iids[i], expand(nvel[i]), expand(vel[i]));
	}
}