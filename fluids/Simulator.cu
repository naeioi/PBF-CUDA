#include "Simulator.h"
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h> 

#include "Simulator_kernel.cuh"

/* Helper functor */

struct helper_duplicate {
	__host__ __device__
	helper_duplicate() {}

	template <typename T> __device__
	thrust::tuple<T, T> operator()(const T t) { return thrust::make_tuple(t, t); }
};

struct getGridxyz {
	float3 llim;
	float h;
	int3 gridDim;
	__host__ __device__
	getGridxyz(const float3 &llim, const int3 &gridDim, float h) : llim(llim), gridDim(gridDim), h(2.f * h) {}

	__device__
	int3 operator()(float3 pos) {
		float3 diff = pos - llim;
		int x = min(max((int)(diff.x / h), 0), gridDim.x - 1), 
			y = min(max((int)(diff.y / h), 0), gridDim.y - 1), 
			z = min(max((int)(diff.z / h), 0), gridDim.z - 1);
		return make_int3(x, y, z);
	}
};

struct xyzToId {
	int3 gridDim;
	__host__ __device__
	xyzToId(const int3 &gridDim) : gridDim(gridDim) {}

	template <typename T> __device__
	int operator()(T x, T y, T z) {
		/*if (x < 0 || x >= gridDim.x || y < 0 || y >= gridDim.y || z < 0 || z >= gridDim.z)
			return -1;*/
		/* TODO */
		/*x = x & 15;
		y = y & 15;
		z = z & 15;*/
		return x * gridDim.y * gridDim.z + y * gridDim.z + z;
	}
};

struct getGridId {

	float3 llim;
	float h;
	int3 gridDim;
	__host__ __device__
	getGridId(const float3 &llim, const int3 &gridDim, float h) : llim(llim), gridDim(gridDim), h(2.f * h) {}

	template <typename T> __device__
	int operator()(T pos) {
		float3 diff = pos - llim;
		int x = diff.x / h, y = diff.y / h, z = diff.z / h;
		x = min(max(x, 0), gridDim.x - 1);
		y = min(max(y, 0), gridDim.y - 1);
		z = min(max(z, 0), gridDim.z - 1);
		return (int)(x * gridDim.y * gridDim.z + y * gridDim.z + z);
	}
};

struct getPoly6 {
	float coef, h2, h9;
	getPoly6(float h) { 
		h2 = h * h;  
		float h3 = h2 * h;
		h9 = h3 * h3;
		coef = 315.f / (64.f * M_PI *  h9);
	}
	__device__
	float operator()(float r2) {
		return coef * (h2 - r2) * (h2 - r2) * (h2 - r2);
	}
};

struct getSpikyGrad {
	float h, h6, coef;
	getSpikyGrad(float h) : h(h) {
		h6 = h * h;
		h6 = h6 * h6 * h6;
		coef = -45.f / (M_PI * h6);
	}

	__device__
	float3 operator()(float3 r) {
		float rlen = length(r);
		return coef * (h - rlen) * (h - rlen) * normalize(r);
	}
};

struct h_updatePosition {
	float3 ulim, llim;
	h_updatePosition(float3 ulim, float3 llim) : ulim(ulim), llim(llim) {}

	template <typename T>
	__device__
	float3 operator()(T t) {
		float3 pos = thrust::get<0>(t), dpos = thrust::get<1>(t);
		pos += dpos;
		/* for now, project particles out of bound onto bounding box surface */
		pos.x = fmaxf(fminf(pos.x, ulim.x - LIM_EPS), llim.x + LIM_EPS);
		pos.y = fmaxf(fminf(pos.y, ulim.y - LIM_EPS), llim.y + LIM_EPS);
		pos.z = fmaxf(fminf(pos.z, ulim.z - LIM_EPS), llim.z + LIM_EPS);

		return pos;
	}
};

struct h_updateVelocity {
	float inv_dt;
	h_updateVelocity(float dt) : inv_dt(1.f / dt) {}

	template <typename T>
	__device__
	float3 operator()(T t) {
		float3 pos = thrust::get<0>(t), npos = thrust::get<1>(t);
		return (npos - pos) * inv_dt;
	}
};

struct DensityBoundary {
	float3 ulim, llim;
	float h;
	DensityBoundary(float3 &ulim, float3 &llim, float h) : ulim(ulim), llim(llim), h(h) {}

	__device__
	float densityAt(float d) {
		if (d > h) return 0.f;
		if (d <= 0.f) return 2 * M_PI / 3;
		return (2 * M_PI / 3) * (h - d) * (h - d) * (h + d);
	}

	__device__
	float operator()(float3 p) {
		return
			densityAt(ulim.x - p.x) +
			densityAt(p.x - llim.x) +
			densityAt(ulim.y - p.y) +
			densityAt(p.y - llim.y) +
			densityAt(ulim.z - p.z) +
			densityAt(p.z - llim.z);
	}
};

/* Intermedia steps */

void Simulator::advect()
{
	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);

	advect_kernel<<<grid_size, block_size>>>(
		dc_iid,
		dc_pos, dc_npos, 
		dc_vel,
		m_nparticle, m_dt, make_float3(0, 0, -m_gravity),
		m_ulim, m_llim);
}

void Simulator::buildGridHash()
{
	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);
	int smem = sizeof(uint) * (block_size + 2);

	thrust::device_ptr<float3> d_pos(dc_pos), d_vel(dc_vel), d_npos(dc_npos), d_nvel(dc_nvel);
	thrust::device_ptr<uint> d_gridId(dc_gridId), d_iid(dc_iid);

	float3 diff = m_ulim - m_llim;
	m_gridHashDim = make_int3((int)ceilf(.5f * diff.x / m_h), (int)ceilf(.5f * diff.y / m_h), (int)ceilf(.5f * diff.z / m_h));
	/* Compute gridId for each particle */
	thrust::transform(
		d_npos, d_npos + m_nparticle,
		d_gridId,
		getGridId(m_llim, m_gridHashDim, m_h));

	/* sort (gridId, pos, vel) by gridId */
	thrust::sort_by_key(
		d_gridId, d_gridId + m_nparticle,
		thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_vel, d_npos, d_nvel, d_iid)));

	/* Compute [gradStart, gradEnd) */
	int cellNum = m_gridHashDim.x * m_gridHashDim.y * m_gridHashDim.z;
	cudaMemset(dc_gridStart, 0, sizeof(dc_gridStart[0]) * cellNum);
	cudaMemset(dc_gridEnd, 0, sizeof(dc_gridEnd[0]) * cellNum);
	computeGridRange<<<grid_size, block_size, smem>>>(dc_gridId, dc_gridStart, dc_gridEnd, m_nparticle);

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeGridRange");

	/*printf("Counter=%d\n", *counter); 
	exit(0);*/
}

void Simulator::correctDensity() 
{
	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);

	// printf("maxStart = %u, maxMin = %u\n", arr[0], arr[1]);

	/// cudaDeviceSynchronize();
	/* dc_npos -> dc_npos */
	computeLambda<<<grid_size, block_size>>>(
		dc_iid,
		dc_lambda, dc_pho,
		dc_gridId, dc_gridStart, dc_gridEnd,
		m_gridHashDim,
		dc_npos, m_nparticle, m_pho0, m_lambda_eps, m_k_boundaryDensity,
		m_h,
		getGridxyz(m_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim), DensityBoundary(m_ulim, m_llim, m_h));

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeLambda");

	m_coef_corr = -m_k_corr / powf(h_poly6(m_h, m_delta_q*m_delta_q), m_n_corr);

	computetpos<<<grid_size, block_size>>>(
		dc_lambda,
		dc_iid,
		dc_gridId, dc_gridStart, dc_gridEnd,
		m_gridHashDim,
		dc_npos, dc_tpos, m_nparticle, m_pho0, m_h, m_coef_corr, m_n_corr,
		getGridxyz(m_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim),
		m_ulim, m_llim);

	thrust::device_ptr<float3> d_npos(dc_npos), d_tpos(dc_tpos);
	thrust::copy_n(d_tpos, m_nparticle, d_npos);
}

void Simulator::correctVelocity() {
	
	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);

	/* XSPH viscosity */
	computeXSPH<<<grid_size, block_size>>>(
		dc_pho,
		dc_iid,
		dc_gridStart, dc_gridEnd, m_gridHashDim,
		dc_npos, dc_vel, dc_nvel, m_nparticle,
		m_c_XSPH, m_h, 
		getGridxyz(m_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim)
		);
}

void Simulator::updateVelocity() {
	/* Warn: assume dc_pos updates to dc_npos after correctDensity() */
	thrust::device_ptr<float3> d_pos(dc_pos), d_npos(dc_npos), d_vel(dc_vel);
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_npos)),
		thrust::make_zip_iterator(thrust::make_tuple(d_pos + m_nparticle, d_npos + m_nparticle)),
		d_vel, h_updateVelocity(m_dt));
}