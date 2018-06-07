#pragma once
#include "helper.h"
#include "GUIParams.h"
#include <math.h>
#include <helper_cuda.h>

class Simulator
{
public:
	Simulator(const GUIParams &params, float3 ulim, float3 llim) : m_ulim(ulim), m_llim(llim) {

		loadParams();
		float3 d = (ulim - llim) / 0.1;
		int ngrid = 2 * (int)(d.x*d.y*d.z);

		checkCudaErrors(cudaMalloc(&dc_gridId, sizeof(uint) * ngrid));
		checkCudaErrors(cudaMalloc(&dc_gridStart, sizeof(uint) * ngrid));
		checkCudaErrors(cudaMalloc(&dc_gridEnd, sizeof(uint) * ngrid));
		checkCudaErrors(cudaMalloc(&dc_lambda, sizeof(float) * MAX_PARTICLE_NUM));
		checkCudaErrors(cudaMalloc(&dc_pho, sizeof(float) * MAX_PARTICLE_NUM));
		checkCudaErrors(cudaMalloc(&dc_tpos, sizeof(float3) * MAX_PARTICLE_NUM));

		/* TODO: can zero initialization be eliminated? This is costly. */
		cudaMemset(dc_gridStart, 0, sizeof(uint) * MAX_PARTICLE_NUM);
		cudaMemset(dc_gridEnd, 0, sizeof(uint) * MAX_PARTICLE_NUM);
	};
	~Simulator() {
		checkCudaErrors(cudaFree(dc_gridId));
		checkCudaErrors(cudaFree(dc_gridStart));
		checkCudaErrors(cudaFree(dc_gridEnd));
		checkCudaErrors(cudaFree(dc_lambda));
		checkCudaErrors(cudaFree(dc_pho));
		checkCudaErrors(cudaFree(dc_tpos));
	}

	/* TODO: may swap(d_pos, d_npos), i.e., the destination is assigned by Simulator, rather than caller */
	void step(uint d_pos, uint d_npos, uint d_vel, uint d_nvel, uint d_iid, int nparticle);

	void loadParams();
	void saveParams();

	void setLim(const float3 &ulim, const float3 &llim);
private:
	void advect();
	void buildGridHash();
	void correctDensity();
	void correctVelocity();
	void updateVelocity();

	float3 *dc_pos, *dc_tpos, *dc_npos, *dc_vel, *dc_nvel;
	uint *dc_iid;
	uint *dc_gridId, *dc_gridStart, *dc_gridEnd;
	float* dc_lambda, *dc_pho;

	float m_gravity, m_h, m_dt, m_pho0, m_lambda_eps, m_delta_q, m_k_corr, m_n_corr, m_k_boundaryDensity, m_c_XSPH;
	float m_coef_corr;
	int m_niter;
	int m_nparticle;
	float3 m_ulim, m_llim;
	int3 m_gridHashDim;
};

