#include "FluidSystem.h"
#include "FixedCubeSource.h"

FluidSystem::FluidSystem()
{
	m_tictoc = 0;

	/* Initialize Component */
	const float
		g = 9.8f,
		h = .1f,
		dt = 0.0083f,
		pho0 = 6378.f,
		lambda_eps = 600.f;
	const float3 ulim = make_float3(.5f, .5f, 1.f), llim = make_float3(-.5f, -.5f, 0.f);
	const int niter = 1;

	m_simulator = new Simulator(g, h, dt, pho0, lambda_eps, niter, ulim, llim);
	m_renderer = new SimpleRenderer(ulim, llim);
	m_source = new FixedCubeSource(
		/* limits */  make_float3(-.5f, .5f, 1.f), make_float3(-.5f+.25f, .5f-.25f, 1.f-.25f), 
		/* numbers */ make_int3(30, 30, 30));
	m_nparticle = 30 * 30 * 30;

	/* Initialize vertex buffer */
	glGenBuffers(1, &d_pos);
	glGenBuffers(1, &d_npos);
	glGenBuffers(1, &d_vel);
	glGenBuffers(1, &d_nvel);
	glGenBuffers(1, &d_iid);
	glGenBuffers(1, &d_niid);

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
	glBindBuffer(GL_ARRAY_BUFFER, d_niid);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLE_NUM * sizeof(uint),   NULL, GL_DYNAMIC_DRAW);
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
		m_nparticle = m_source->update(d_npos, d_nvel, d_niid, MAX_PARTICLE_NUM);
}

void FluidSystem::stepSimulate() {
	if (!m_tictoc)
		m_simulator->step(d_pos, d_npos, d_vel, d_nvel, d_iid, d_niid, m_nparticle);
	else 
		m_simulator->step(d_npos, d_pos, d_nvel, d_vel, d_niid, d_iid, m_nparticle);
	
	m_tictoc = !m_tictoc;
}

void FluidSystem::render() {
	if (!m_tictoc)
		m_renderer->render(d_pos, m_nparticle);
	else
		m_renderer->render(d_npos, m_nparticle);
}

FluidSystem::~FluidSystem()
{
	if (m_renderer) delete m_renderer;
	if (m_simulator) delete m_simulator;
	if (m_source) delete m_source;
}
