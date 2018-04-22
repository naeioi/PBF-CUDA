#pragma once

struct FluidParams {
	int niter;
	float pho0;
	float g;
	float h;
	float dt;
	float lambda_eps;
	float delta_q;
	float k_corr;
	float n_corr;
	float k_boundaryDensity;
	float c_XSPH;
};