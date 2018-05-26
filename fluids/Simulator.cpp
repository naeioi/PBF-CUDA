#include "Simulator.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <glad\glad.h>
// #include <GLFW\glfw3.h>
#include <cuda_gl_interop.h>

void Simulator::step(uint d_pos, uint d_npos, uint d_vel, uint d_nvel, uint d_iid, int nparticle) 
{
	m_nparticle = nparticle;

	struct cudaGraphicsResource *dcr_pos, *dcr_npos;
	struct cudaGraphicsResource *dcr_vel, *dcr_nvel;
	struct cudaGraphicsResource *dcr_iid;

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&dcr_pos, d_pos, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&dcr_vel, d_vel, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&dcr_iid, d_iid, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&dcr_npos, d_npos, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&dcr_nvel, d_nvel, cudaGraphicsMapFlagsNone));

	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_pos, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_vel, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_iid, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_npos, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_nvel, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dc_pos, &size, dcr_pos));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dc_vel, &size, dcr_vel));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dc_iid, &size, dcr_iid));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dc_npos, &size, dcr_npos));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dc_nvel, &size, dcr_nvel));

	/* Simulate logic */

	/* Real upper and lowe limit after advection */
	static int round = 0;
	fprintf(stderr, "-- Round %d --\n", round++);
	advect();
	
	cudaDeviceSynchronize();
	printf("advect() done.\n");

	buildGridHash();

	cudaDeviceSynchronize();
	printf("buildGridHash() done.\n");

	for (uint i = 0; i < m_niter; i++) {
		printf("== Iter %d ==\n", i);
		correctDensity();
		cudaDeviceSynchronize();
	}

	/* update Velocity */
	updateVelocity();
	correctVelocity();

	cudaDeviceSynchronize();
	//exit(0);

	/* Simulate logic ends */
	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_pos, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_vel, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_iid, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_npos, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_nvel, 0));

	/**
	 * Omitting unregister resource will cause memory leak
	 * Program will be blown away after several iterations.
	 */
	checkCudaErrors(cudaGraphicsUnregisterResource(dcr_pos));
	checkCudaErrors(cudaGraphicsUnregisterResource(dcr_vel));
	checkCudaErrors(cudaGraphicsUnregisterResource(dcr_iid));
	checkCudaErrors(cudaGraphicsUnregisterResource(dcr_npos));
	checkCudaErrors(cudaGraphicsUnregisterResource(dcr_nvel));
}

void Simulator::loadParams()
{
	const GUIParams & params = GUIParams::getInstance();
	m_dt = params.dt;
	m_gravity = params.g;
	m_h = params.h;
	m_pho0 = params.pho0;
	m_lambda_eps = params.lambda_eps;
	m_delta_q = params.delta_q;
	m_k_corr = params.k_corr;
	m_n_corr = params.n_corr;
	m_k_boundaryDensity = params.k_boundaryDensity;
	m_c_XSPH = params.c_XSPH;
	m_niter = params.niter;
}

void Simulator::saveParams() {
	GUIParams &params = GUIParams::getInstance();
	params.dt = m_dt;
	params.g = m_gravity;
	params.h = m_h;
	params.pho0 = m_pho0;
	params.lambda_eps = m_lambda_eps;
	params.delta_q = m_delta_q;
	params.k_corr = m_k_corr;
	params.n_corr = m_n_corr;
	params.k_boundaryDensity = m_k_boundaryDensity;
	params.c_XSPH = m_c_XSPH;
	params.niter = m_niter;
}
