#pragma once
#include "Camera.h"

class SSFRenderer
{
public:
	SSFRenderer(Camera *camera, int width, int height);
	~SSFRenderer();

	void destroy();

	void render(uint p_vao, int nparticle);

private:
};

