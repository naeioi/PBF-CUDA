#include "FluidSystem.h"
#include "FixedCubeSource.h"
#include "DoubleDamSource.h"
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
	params.shading_option = GUIParams::Full;
	params.keep_edge = 1;
	params.blur_option = 0;

	m_ulim = make_float3(2.f, 2.f, 4.f);
	m_llim = make_float3(-2.f, -2.f, 0.f);
	m_A_llim = make_float3(0.f, 0.f, 0.f);
	m_A_ulim = make_float3(2.f, 0.f, 0.f);
	m_w = 0.05;
	// const float3 ulim = make_float3(1.f, 1.f, 4.f), llim = make_float3(-2.f, -2.f, 0.f);
	// const float3 ulim = make_float3(1.f, 1.f, 4.f), llim = make_float3(-2.f, -2.f, 0.f);
	const glm::vec3 cam_pos(1.f, -5.f, 2.f), cam_focus(0, 0, 1.5f);

	m_simulator = new Simulator(params, m_ulim, m_llim);
	m_renderer = new Renderer(cam_pos, cam_focus, m_ulim, m_llim, [this]() { m_nextFrame = true; });

	/* Single cube */
	//float dd = 1.f / 20;
	//float d1 = dd * 30, d2 = dd * 30, d3 = dd * 30;
	//m_source = new FixedCubeSource(
	//	/* limits */  make_float3(1.8f, 1.8f, 3.8f), make_float3(1.8f-d1, 1.8f-d2, 3.8f-d3),
	//	/* numbers */ make_int3(30, 30, 30));
	//m_nparticle = 30 * 30 * 30;

	/* Double cube */
	float dd = 1.f / 20;
	float d1 = dd * 20, d2 = dd * 20, d3 = dd * 40;
	m_source = new DoubleDamSource(
		make_float3(-1.8f, 1.8f, 3.8f), make_float3(-1.8f+d1, 1.8f-d2, 3.8f-d3), make_int3(20, 20, 40),
		make_float3(1.8f-d1, -1.8f+d2, 3.8f), make_float3(1.8f, -1.8f, 3.8f-d3), make_int3(20, 20, 40));
		
	m_nparticle = 2 * 20 * 20 * 40;

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

	/* Sweep boundary */
	auto &input = Input::getInstance();
	if (input.moving) {
		float t = m_w * (input.frameCount - input.startMovingFrame);
		float phi = sin(t);
		m_simulator->setLim(m_ulim + m_A_ulim * phi, m_llim + m_A_llim * phi);
	}

	if (!m_tictoc)
		m_simulator->step(d_pos, d_npos, d_vel, d_nvel, d_iid, m_nparticle);
	else 
		m_simulator->step(d_npos, d_pos, d_nvel, d_vel, d_iid, m_nparticle);
	
	m_tictoc = !m_tictoc;
	m_nextFrame = false;
	m_renderer->m_input->frameCount++;
}

void FluidSystem::render() {
	/* Sweep boundary */
	auto &input = Input::getInstance();
	if (input.moving) {
		float t = m_w * (input.frameCount - input.startMovingFrame);
		float phi = sin(t);
		m_renderer->setLim(m_ulim + m_A_ulim * phi, m_llim + m_A_llim * phi);
	}

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
