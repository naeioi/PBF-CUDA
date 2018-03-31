#pragma once
#include <GLFW\glfw3.h>
#include "Simulator.h"
#include "SimpleRenderer.h"
#include "ParticleSource.h"

class FluidSystem
{
public:

	FluidSystem();
	~FluidSystem();

	void stepSimulate();
	void render();

	void setFluidSource();
	void setDt(float);

	void reset();
private:

	/* Working components */
	ParticleSource *m_source;
	Simulator *m_simulator;
	SimpleRenderer  *m_renderer;

	/* Parameters */
	float s_dt;
	float s_h;

	/* Particles states */

	/* position: float3 GLBuffer */
	GLuint d_pos, d_npos;
	/* velocity: float3 GLBuffer */
	GLuint d_vec, d_nvec;
	/* initial id: uint GLBuffer */
	GLuint d_iid, d_niid;
};

