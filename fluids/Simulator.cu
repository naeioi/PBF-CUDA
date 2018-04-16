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
	T operator()(T a, T b) { 
		float3 &amax = thrust::get<0>(a), &amin = thrust::get<1>(a),
			   &bmax = thrust::get<0>(b), &bmin = thrust::get<1>(b);
		return thrust::make_tuple(
			make_float3(fmaxf(amax.x, bmax.x), fmaxf(amax.y, bmax.y), fmaxf(amax.z, bmax.z)),
			make_float3(fminf(amin.x, bmin.x), fminf(amin.y, bmin.y), fminf(amin.z, bmin.z)));
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
		// printf("(%f,%f,%f)\n", diff.x, diff.y, diff.z);
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
		float3 pos = thrust::get<0>(t), dpos = thrust::get<1>(t);
		pos += dpos;
		/* for now, project particles out of bound onto bounding box surface */
		pos.x = fmaxf(fminf(pos.x, ulim.x), llim.x);
		pos.y = fmaxf(fminf(pos.y, ulim.y), llim.y);
		pos.z = fmaxf(fminf(pos.z, ulim.z), llim.z);

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
		// printf("(%f,%f,%f) -> (%f,%f,%f)\n", pos.x, pos.y, pos.z, npos.x, npos.y, npos.z);
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
		m_nparticle, m_dt, make_float3(0, 0, -m_gravity),
		m_ulim, m_llim);

	/* find real limits after advection */
	auto t = thrust::transform_reduce(
		thrust::device_ptr<float3>(dc_npos), thrust::device_ptr<float3>(dc_npos + m_nparticle),
		helper_duplicate(),
		thrust::make_tuple(m_ulim, m_llim),
		getExtrema());

	 m_real_ulim = thrust::get<0>(t) + m_h * LIM_EPS;
	 m_real_llim = thrust::get<1>(t) - m_h * LIM_EPS;
	 // fprintf(stderr, "m_real_ulim(%f,%f,%f)\n", m_real_ulim.x, m_real_ulim.y, m_real_ulim.z);
	 // fprintf(stderr, "m_real_llim(%f,%f,%f)\n", m_real_llim.x, m_real_llim.y, m_real_llim.z);
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

	thrust::device_ptr<float3> d_pos(dc_pos), d_vel(dc_vel), d_npos(dc_npos), d_nvel(dc_nvel);
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
		thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_vel, d_npos, d_nvel)));

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: before computeGridRange");

	/* Compute [gradStart, gradEnd) */
	// int *counter;
	// cudaMallocManaged(&counter, sizeof(counter));
	computeGridRange<<<grid_size, block_size, smem>>>(dc_gridId, dc_gridStart, dc_gridEnd, m_nparticle);

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeGridRange");

	/*printf("Counter=%d\n", *counter); 
	exit(0);*/
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

	// printf("maxStart = %u, maxMin = %u\n", arr[0], arr[1]);

	/// cudaDeviceSynchronize();
	/* dc_npos -> dc_npos */
	computeLambda<<<grid_size, block_size>>>(
		dc_lambda, /*dc_gradl2,*/
		dc_gridId, dc_gridStart, dc_gridEnd,
		m_gridHashDim,
		dc_npos, m_nparticle, m_pho0, m_lambda_eps,
		/* getPoly6(m_h), getSpikyGrad(m_h), */ m_h,
		getGridxyz(m_real_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim));

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeLambda");

	computedpos<<<grid_size, block_size>>>(
		dc_lambda, /*dc_gradl2,*/
		dc_gridId, dc_gridStart, dc_gridEnd,
		m_gridHashDim,
		dc_npos, dc_dpos, m_nparticle, m_pho0, m_h,
		getGridxyz(m_real_llim, m_gridHashDim, m_h), xyzToId(m_gridHashDim));

	/* update position */
	thrust::device_ptr<float3> d_npos(dc_npos), d_dpos(dc_dpos);
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(d_npos, d_dpos)),
		thrust::make_zip_iterator(thrust::make_tuple(d_npos+m_nparticle, d_dpos+m_nparticle)),
		d_npos, h_updatePosition(m_ulim, m_llim));

	// cudaDeviceSynchronize();
	// getLastCudaError("Kernel execution failed: computeLambda");
}

void Simulator::updateVelocity() {
	/* Warn: assume dc_pos updates to dc_npos after correctDensity() */
	thrust::device_ptr<float3> d_pos(dc_pos), d_npos(dc_npos), d_nvel(dc_nvel);
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_npos)),
		thrust::make_zip_iterator(thrust::make_tuple(d_pos + m_nparticle, d_npos + m_nparticle)),
		d_nvel, h_updateVelocity(m_dt));
}