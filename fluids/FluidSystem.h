#pragma once
#include "Simulator.h"
#include "SimpleRenderer.h"
#include "ParticleSource.h"

#include <GLFW\glfw3.h>

class FluidSystem
{
public:

	FluidSystem();
	~FluidSystem();

	void initSource();
	void stepSource();
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
	GLuint d_vel, d_nvel;
	/* initial id: uint GLBuffer */
	GLuint d_iid, d_niid;

	/* tic: d_pos, toc: d_npos for rendering */
	bool m_tictoc, m_nextFrame;
	int  m_nparticle;
};

