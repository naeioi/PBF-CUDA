#pragma once
#include "Shader.h"
#include "Camera.h"
#include "helper.h"

struct SSFRendererImpl
{
	enum { A = 0, B = 1 };

	SSFRendererImpl(Camera *camera, int width, int height, uint sky_texture);

	void destroy();

	void render(uint p_vao, int nparticle);
	void renderDepth();
	void renderThick();
	void restoreNormal();
	void computeH();
	void updateDepth();
	void smoothDepth();
	void shading();

	void loadParams();

	int m_niter;
	int m_width, m_height;
	/* particle vertex array object */
	uint p_vao;
	int m_nparticle;
	/* Depth update ratio */
	float m_k; 

	uint m_quad_vao;

	/* framebuffer fluid rendered to */
	uint d_fbo;
	/* Pingpong flag */
	bool m_ab; 
	/* Depth texture of type GL_DEPTH_COMPONENT32F */
	uint d_depth;
	/* Pingpong depth texture of type GL_RED32F */
	uint d_depth_a, d_depth_b;
	/* normal & D in GL_RGBA32F */
	uint d_normal_D;
	/* curvature */
	uint d_H;
	/* Thickness */
	uint d_thick;
	/* Cuda resources to map/unmap texture */
	//struct cudaGraphicsResource *dcr_depth, *dcr_normal_D, *dcr_H;
	inline uint zTex1() { return m_ab ? d_depth_a : d_depth_b;  } /* Smooth source */
	inline uint zTex2() { return m_ab ? d_depth_b : d_depth_a;  } /* Render source */

	/*texture<float, cudaTextureType2D, cudaReadModeElementType> *dc_depth, *dc_H;
	texture<float4, cudaTextureType2D, cudaReadModeElementType> *dc_normal_D;*/

	/* Helper function to map/unmap above textures */
	void mapResources();
	void unmapResources();

	Shader *m_s_get_depth, *m_s_get_thick, *m_s_shading;
	Shader *m_s_restore_normal, *m_s_computeH, *m_s_update_depth;

	/* Bilateral filter */
	float m_blur_r, m_blur_z;
	int m_kernel_r;
	Shader *m_s_smooth_depth;

	/* Schlick's approximation on Fresnel law */
	float m_r0;
	uint d_sky;

	Camera *m_camera;
	ProjectionInfo m_pi;
};

