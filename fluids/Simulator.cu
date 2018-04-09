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

struct getExtrema {
	typedef thrust::pair<float3, float3> PFF;

	__host__ __device__
	getExtrema() {}

	template <typename T> __device__
	T operator()(T acc_, T p_) { 
		float3 acc_ulim = thrust::get<0>(acc_), acc_llim = thrust::get<1>(acc_);
		float3 p = thrust::get<0>(p_);
		return thrust::make_tuple(
			make_float3(max(acc_ulim.x, p.x), max(acc_ulim.y, p.y), max(acc_ulim.z, p.z)),
			make_float3(min(acc_llim.x, p.x), min(acc_llim.y, p.y), min(acc_llim.z, p.z)));
	}
};

struct getGridxyz {
	float3 llim;
	float h;
	uint3 gridDim;
	__host__ __device__
	getGridxyz(const float3 &llim, const uint3 &gridDim, float h) : llim(llim), gridDim(gridDim), h(2.f*h) {}

	__device__
	int3 operator()(float3 pos) {
		float3 diff = pos - llim;
		int x = diff.x / h, y = diff.y / h, z = diff.z / h;
		return make_int3(x, y, z);
	}
};

struct xyzToId {
	uint3 gridDim;
	__host__ __device__
	xyzToId(const uint3 &gridDim) : gridDim(gridDim) {}

	template <typename T> __device__
	int operator()(T x, T y, T z) {
		/*if (x < 0 || x >= gridDim.x || y < 0 || y >= gridDim.y || z < 0 || z >= gridDim.z)
			return -1;*/
		/* TODO */
		x = x & 15;
		y = y & 15;
		z = z & 15;
		return x * gridDim.y * gridDim.z + y * gridDim.z + z;
	}
};

struct getGridId {

	float3 llim;
	float h;
	uint3 gridDim;
	__host__ __device__
		getGridId(const float3 &llim, const uint3 &gridDim, float h) : llim(llim), gridDim(gridDim), h(2.f*h) {}

	template <typename T> __device__
		uint operator()(T pos) {
		float3 diff = pos - llim;
		int x = diff.x / h, y = diff.y / h, z = diff.z / h;
		return (uint)(x * gridDim.y * gridDim.z + y * gridDim.z + z);
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
		float3 pos = thrust::get<0>(t), grad = thrust::get<1>(t);
		float lambda = thrust::get<2>(t);
		pos += lambda * grad;
		/* for now, project particles out of bound onto bounding box surface */
		pos.x = max(min(pos.x, ulim.x), llim.x);
		pos.y = max(min(pos.y, ulim.y), llim.y);
		pos.z = max(min(pos.z, ulim.z), llim.z);

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

/* Intermedia steps */

void Simulator::advect()
{
	/* Input:  dc_pos,  dc_vel,  dc_iid
	 * Output: dc_npos, dc_nvel, dc_niid
	 */

	/* TODO: 1. try different block_size. 2. try cudaOccupancyMaxPotentialBlockSize() */
	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);

	advect_kernel<<<grid_size, block_size>>>(
		dc_pos, dc_npos, 
		dc_vel, dc_nvel, 
		m_nparticle, m_dt, make_float3(0, 0, -m_gravity));

	/* find real limits after advection */
	auto t = thrust::transform_reduce(
		thrust::device_ptr<float3>(dc_npos), thrust::device_ptr<float3>(dc_npos + m_nparticle),
		helper_duplicate(),
		thrust::make_tuple(m_ulim, m_llim),
		getExtrema());

	 m_real_ulim = thrust::get<0>(t) + m_h * LIM_EPS;
	 m_real_llim = thrust::get<1>(t) - m_h * LIM_EPS;
}

void Simulator::buildGridHash()
{
	/* Input:  dc_npos, dc_nvel, dc_niid
	 * Output: dc_npos, dc_nvel, dc_niid. Sorted by GridId(dc_npos)
	 *         dc_gridStart, dc_gridEnd.  In start closed end open range.
	 * Intermedia:
	 *         dc_gridId
	 */

	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);
	int smem = sizeof(uint) * (block_size + 2);

	thrust::device_ptr<float3> d_npos(dc_npos), d_nvel(dc_nvel);
	thrust::device_ptr<uint> d_gridId(dc_gridId);

	/* Compute real_ulim and real_llim */
	computeGridHashDim();
	/* Compute gridId for each particle */
	thrust::transform(
		d_npos, d_npos + m_nparticle,
		d_gridId,
		getGridId(m_real_llim, m_gridHashDim, m_h));

	/* sort (gridId, pos, vel) by gridId */
	thrust::sort_by_key(
		d_gridId, d_gridId + m_nparticle,
		thrust::make_zip_iterator(thrust::make_tuple(d_npos, d_nvel)));

	/* Compute [gradStart, gradEnd) */
	computeGridRange<<<grid_size, block_size, smem>>>(dc_gridId, dc_gridStart, dc_gridEnd, m_nparticle);

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeGridRange");
}

void Simulator::computeGridHashDim() {
	/* Input: m_real_ulim, m_real_llim,
	 * Output: m_gridHashDim
	 */
	float3 diff = m_real_ulim - m_real_llim;
	m_gridHashDim = make_uint3((uint)ceilf(diff.x / m_h), (uint)ceilf(diff.y / m_h), (uint)ceilf(diff.z / m_h));
}

void Simulator::correctDensity() 
{
	/* Input:  
	 */

	int block_size = 256;
	int grid_size = ceilDiv(m_nparticle, block_size);

	uint* arr;
	cudaMallocManaged(&arr, sizeof(uint) * 3);

	// printf("maxStart = %u, maxMin = %u\n", arr[0], arr[1]);

	/// cudaDeviceSynchronize();
	/* dc_npos -> dc_npos */
	computeLambda<<<grid_size, block_size>>>(
		dc_lambda, dc_grad,
		dc_gridId, dc_gridStart, dc_gridEnd,
		m_gridHashDim,
		dc_npos, m_nparticle, m_pho0, m_lambda_eps,
		/* getPoly6(m_h), getSpikyGrad(m_h), */ m_h,
		getGridxyz(m_real_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim),
		&arr[0], &arr[1]);

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computeLambda");

	// printf("maxStart = %u, maxMin = %u\n", arr[0], arr[1]);
	// printf("\n");

	/* update position */
	thrust::device_ptr<float3> d_npos(dc_npos), d_grad(dc_grad);
	thrust::device_ptr<float> d_lambda(dc_lambda);
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(d_npos, d_grad, d_lambda)),
		thrust::make_zip_iterator(thrust::make_tuple(d_npos+m_nparticle, d_grad+m_nparticle, d_lambda+m_nparticle)),
		d_npos, h_updatePosition(m_ulim, m_llim));

	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed: computeLambda");
}

void Simulator::updateVelocity() {
	/* Warn: assume dc_pos updates to dc_npos after correctDensity() */
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(dc_pos, dc_npos)),
		thrust::make_zip_iterator(thrust::make_tuple(dc_pos + m_nparticle, dc_npos + m_nparticle)),
		dc_nvel, h_updateVelocity(m_dt));
}