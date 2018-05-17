#include "SSFRendererImpl.h"
#include "SSFRenderer.h"

SSFRenderer::SSFRenderer(Camera *camera, int width, int height)
{
	m_impl = new SSFRendererImpl(camera, width, height);
}


SSFRenderer::~SSFRenderer()
{
	m_impl->destroy();
}

void SSFRenderer::destroy()
{
	m_impl->destroy();
}

void SSFRenderer::render(uint p_vao, int nparticle)
{
	m_impl->render(p_vao, nparticle);
}
