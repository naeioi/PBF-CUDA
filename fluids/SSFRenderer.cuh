#pragma once
#include "helper.h"
#include "Shader.h"
#include "Camera.h"
#include <GLFW\glfw3.h>

class SSFRenderer
{
public:
	SSFRenderer(Camera *camera, int width, int height);
	~SSFRenderer();

	// void initialize(uint p_vao, int nparticle);
	void destroy();

	void render(uint p_vao, int nparticle);

private: 

	void __render();

	int m_width, m_height;
	/* particle vertex array object */
	uint p_vao;
	int m_nparticle;

	uint m_quad_vao;

	/* framebuffer fluid rendered to */
	uint d_fbo;
	/* depth / pos / normal texture */
	uint d_depth, d_pos, d_normal;
	texture<float, cudaTextureType2D, cudaReadModeElementType> *dc_depth;
	texture<float3, cudaTextureType2D, cudaReadModeElementType> *dc_pos, *dc_normal;

	Shader *m_s_get_depth, *m_s_put_depth;
	Camera *m_camera;
	ProjectionInfo m_pi;
};

