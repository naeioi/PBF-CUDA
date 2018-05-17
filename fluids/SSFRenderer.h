#pragma once
#include "helper.h"
#include "Camera.h"

struct SSFRendererImpl;

class SSFRenderer
{
public:
	SSFRenderer(Camera *camera, int width, int height);
	~SSFRenderer();

	void destroy();
	void render(uint p_vao, int nparticle);

private:
	SSFRendererImpl *m_impl;
};

