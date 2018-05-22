#pragma once
#include "Shader.h"
#include "Camera.h"
#include "helper.h"

struct SSFRendererImpl
{
	SSFRendererImpl(Camera *camera, int width, int height);

	void destroy();

	void render(uint p_vao, int nparticle);
	void renderDepth();
	void restoreNormal();
	void computeH();
	void updateDepth();
	void renderPlane();

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
	/* depth / pos / normal texture */
	uint d_depth, d_depth_r, d_normal_D, d_H;
	/* Cuda resources to map/unmap texture */
	//struct cudaGraphicsResource *dcr_depth, *dcr_normal_D, *dcr_H;

	/*texture<float, cudaTextureType2D, cudaReadModeElementType> *dc_depth, *dc_H;
	texture<float4, cudaTextureType2D, cudaReadModeElementType> *dc_normal_D;*/

	/* Helper function to map/unmap above textures */
	void mapResources();
	void unmapResources();

	Shader *m_s_get_depth, *m_s_put_depth;
	Shader *m_s_restore_normal, *m_s_computeH, *m_s_update_depth;
	Camera *m_camera;
	ProjectionInfo m_pi;

};

