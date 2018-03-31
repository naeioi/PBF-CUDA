#include "FluidSystem.h"

FluidSystem::FluidSystem()
{
	m_simulator = new Simulator();
	m_renderer = new SimpleRenderer();
	m_source = new ParticleSource();
}


FluidSystem::~FluidSystem()
{
	if (m_renderer) delete m_renderer;
	if (m_simulator) delete m_simulator;
	if (m_source) delete m_source;
}
