#include "FluidSystem.h"
#include "FixedCubeSource.h"
#include "GUIParams.h"
#include <glm/common.hpp>
using namespace glm;

FluidSystem::FluidSystem()
{
	m_tictoc = 0;
	m_nextFrame = false;

	/* Initialize Component */
	GUIParams &params = GUIParams::getInstance();
	params.g = 9.8f,
	params.h = .1f,
	params.dt = 0.0083f,
	params.pho0 = 8000.f,
	params.lambda_eps = 1000.f,
	params.delta_q = 0.3 * params.h,
	params.k_corr = 0.001f,
	params.n_corr = 4,
	params.k_boundaryDensity = 0.f,
	params.c_XSPH = 0.5f;
	params.niter = 4;
	params.smooth_niter = 2;
	params.kernel_r = 10.f;
	params.sigma_r = 6.f;
	params.sigma_z = 0.1f;
	params.shading_option = 0;

	const float3 ulim = make_float3(2.f, 2.f, 2.f), llim = make_float3(-2.f, -2.f, 0.f);
	const glm::vec3 cam_pos(1.f, -5.f, 2.f), cam_focus(0, 0, 1.5f);

	m_simulator = new Simulator(params, ulim, llim);
	m_renderer = new SimpleRenderer(cam_pos, cam_focus, ulim, llim, [this]() { m_nextFrame = true; });
	m_source = new FixedCubeSource(
		/* limits */  make_float3(.5f, .5f, 1.8f), make_float3(-.5f, -.5f, .8f),
		/* numbers */ make_int3(40, 40, 20));
	m_nparticle = 40 * 40 * 20;

	/* Initialize vertex buffer */
	glGenBuffers(1, &d_pos);
	glGenBuffers(1, &d_npos);
	glGenBuffers(1, &d_vel);
	glGenBuffers(1, &d_nvel);
	glGenBuffers(1, &d_iid);

	glBindBuffer(GL_ARRAY_BUFFER, d_pos);
	glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
	checkGLErr();
	glBindBuffer(GL_ARRAY_BUFFER, d_npos);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
	checkGLErr();
	glBindBuffer(GL_ARRAY_BUFFER, d_vel);
	glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
	checkGLErr();
	glBindBuffer(GL_ARRAY_BUFFER, d_nvel);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
	checkGLErr();
	glBindBuffer(GL_ARRAY_BUFFER, d_iid);
	glBufferData(GL_ARRAY_BUFFER,  MAX_PARTICLE_NUM * sizeof(uint),   NULL, GL_DYNAMIC_DRAW);
	checkGLErr();
}

void FluidSystem::initSource() 
{
	m_source->initialize(d_pos, d_vel, d_iid, MAX_PARTICLE_NUM);
}

void FluidSystem::stepSource() {
	if (!m_tictoc)
		m_nparticle = m_source->update(d_pos,  d_vel,  d_iid,  MAX_PARTICLE_NUM);
	else 
		m_nparticle = m_source->update(d_npos, d_nvel, d_iid, MAX_PARTICLE_NUM);
}

void FluidSystem::stepSimulate() {
	if (!(m_renderer->m_input->running || m_nextFrame)) return;

	m_simulator->loadParams();

	if (!m_tictoc)
		m_simulator->step(d_pos, d_npos, d_vel, d_nvel, d_iid, m_nparticle);
	else 
		m_simulator->step(d_npos, d_pos, d_nvel, d_vel, d_iid, m_nparticle);
	
	m_tictoc = !m_tictoc;
	m_nextFrame = false;
	m_renderer->m_input->frameCount++;
}

void FluidSystem::render() {
	bool t = m_tictoc ^ m_renderer->m_input->lastFrame;
	if (!t)
		m_renderer->render(d_pos, d_iid, m_nparticle);
	else
		m_renderer->render(d_npos, d_iid, m_nparticle);
}

FluidSystem::~FluidSystem()
{
	if (m_renderer) delete m_renderer;
	if (m_simulator) delete m_simulator;
	if (m_source) delete m_source;
}
