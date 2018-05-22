#include "helper.h"
#include "SSFRendererImpl.h"
#include <GLFW\glfw3.h>
#include <glad\glad.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

static float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
    // positions   // texCoords
    -1.0f,  1.0f,  0.0f, 1.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
    1.0f, -1.0f,  1.0f, 0.0f,

    -1.0f,  1.0f,  0.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 0.0f,
    1.0f,  1.0f,  1.0f, 1.0f
};

SSFRendererImpl::SSFRendererImpl(Camera *camera, int width, int height)
{
	m_k = 1.f;
	m_niter = 1;

	/* TODO: consider how to handle resolution change */
	this->m_camera = camera;
	this->m_width = width;
	this->m_height = height;
	this->m_pi = camera->getProjectionInfo();

	/* Allocate depth / normal_D / H texture */
	glGenTextures(1, &d_depth);
	glGenTextures(1, &d_depth_r);
	glGenTextures(1, &d_normal_D);
	glGenTextures(1, &d_H);

	glBindTexture(GL_TEXTURE_2D, d_normal_D);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	/* TODO: check effect of GL_NEAREST */
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_depth_r);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_H);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();

	/* TODO: Bind texture to CUDA resource */
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_normal_D, d_normal_D, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	/* CUDA does not support interop with GL_DEPTH_COMPONENT texture ! 
	 * As a workaround, first render to a depth texture (d_depth)
	 * then copy depth texture to a color texture (d_depth_r), which contains only red channel 
	 */
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_depth, d_depth_r, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_H, d_H, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	/* Allocate framebuffer & Binding depth texture */
	glGenFramebuffers(1, &d_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, d_depth, 0);
	
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, d_depth_r, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, d_normal_D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, d_H, 0);

	checkFramebufferComplete();
	checkGLErr();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Load shaders */
	m_s_get_depth = new Shader(Filename("SSFget_depth.v.glsl"), Filename("SSFget_depth.f.glsl"));
	m_s_put_depth = new Shader(Filename("SSFput_depth.v.glsl"), Filename("SSFput_depth.f.glsl"));
	m_s_restore_normal = new Shader(Filename("SSFrestore_normal.v.glsl"), Filename("SSFrestore_normal.f.glsl"));
	m_s_computeH = new Shader(Filename("SSFcomputeH.v.glsl"), Filename("SSFcomputeH.f.glsl"));
	m_s_update_depth = new Shader(Filename("SSFupdate_depth.v.glsl"), Filename("SSFupdate_depth.f.glsl"));

	/* Load quad vao */
	uint quad_vbo;
	glGenVertexArrays(1, &m_quad_vao);
	glGenBuffers(1, &quad_vbo);
	glBindVertexArray(m_quad_vao);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void SSFRendererImpl::destroy() {
	// if (!dc_depth) return;
	/* TODO */
}

void SSFRendererImpl::renderDepth() {
	/* After renderDepth(), z_c is store at d_depth 
	 * Linearize depth (z_e) is stored at d_depth_r  
	 */

	/* Render to framebuffer */
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glDisable(GL_BLEND);

	/* Reset depth_r to maximum */
	GLfloat red[] = { 100.f };
	glClearTexImage(d_depth_r, 0, GL_RED, GL_FLOAT, red);
	checkGLErr();

	/* Have to assign COLOR_ATTACHMENT0 to first drawbuffer
	 * because later we assign COLOR_ATTACHMENT2 to first drawbuffer
	 */
	GLenum bufs[] = { GL_COLOR_ATTACHMENT0 /* d_depth_r */ };
	glDrawBuffers(1, bufs);

	m_s_get_depth->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_get_depth->setUnif("s_h", m_height);
	m_s_get_depth->setUnif("p_t", i.t);
	m_s_get_depth->setUnif("p_n", i.n);
	m_s_get_depth->setUnif("p_f", i.f);
	m_s_get_depth->setUnif("r", 0.1f * 0.2f);
	m_s_get_depth->setUnif("pointRadius", 50.f);

	glEnable(GL_DEPTH_TEST);
	glBindVertexArray(p_vao);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		fexit(-1, "Framebuffer not complete\n");

	glClear(GL_DEPTH_BUFFER_BIT);
	glDrawArrays(GL_POINTS, 0, m_nparticle);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}

void SSFRendererImpl::renderPlane() {

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Draw depth in greyscale */
	m_s_put_depth->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_put_depth->setUnif("p_n", i.n);
	m_s_put_depth->setUnif("p_f", i.f);
	m_s_put_depth->setUnif("p_t", i.t);
	m_s_put_depth->setUnif("p_r", i.r);

	glEnable(GL_DEPTH_TEST);
	glBindVertexArray(m_quad_vao);
	glBindTexture(GL_TEXTURE_2D, d_depth_r);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	// glDisable(GL_DEPTH_TEST);
}

void SSFRendererImpl::render(uint p_vao, int nparticle) {

	this->p_vao = p_vao;
	this->m_nparticle = nparticle;

	renderDepth();

	// mapResources();
	for (int i = 0; i < m_niter; i++) {
		restoreNormal();
		computeH();
		// updateDepth();
	}
	// unmapResources();

	renderPlane();
}

//void SSFRendererImpl::mapResources() {
//	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_depth, 0));
//	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_normal_D, 0));
//	checkCudaErrors(cudaGraphicsMapResources(1, &dcr_H, 0));
//
//	size_t size;
//	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)dc_depth, &size, dcr_depth));
//	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)dc_normal_D, &size, dcr_normal_D));
//	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)dc_H, &size, dcr_H));
//}
//
//void SSFRendererImpl::unmapResources() {
//	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_depth, 0));
//	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_normal_D, 0));
//	checkCudaErrors(cudaGraphicsUnmapResources(1, &dcr_H, 0));
//
//	/* TODO: check if need unregister resource using cudaGraphicsUnregisterResource() */
//}

void SSFRendererImpl::restoreNormal() {

	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glDisable(GL_BLEND);

	m_s_restore_normal->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_restore_normal->setUnif("p_n", i.n);
	m_s_restore_normal->setUnif("p_f", i.f);
	m_s_restore_normal->setUnif("p_t", i.t);
	m_s_restore_normal->setUnif("p_r", i.r);
	m_s_restore_normal->setUnif("s_w", (float)m_width);
	m_s_restore_normal->setUnif("s_h", (float)m_height);

	glDisable(GL_DEPTH_TEST);

	GLenum bufs[] = { GL_COLOR_ATTACHMENT1 /* d_normal_D */ };
	glDrawBuffers(1, bufs);
	
	glBindVertexArray(m_quad_vao);
	glBindTexture(GL_TEXTURE_2D, d_depth_r);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}

void SSFRendererImpl::computeH() {
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glDisable(GL_BLEND);

	m_s_computeH->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_computeH->setUnif("p_n", i.n);
	m_s_computeH->setUnif("p_f", i.f);
	m_s_computeH->setUnif("p_t", i.t);
	m_s_computeH->setUnif("p_r", i.r);
	m_s_computeH->setUnif("s_w", (float)m_width);
	m_s_computeH->setUnif("s_h", (float)m_height);

	glDisable(GL_DEPTH_TEST);

	GLenum bufs[] = { GL_COLOR_ATTACHMENT2 /* d_H */ };
	glDrawBuffers(1, bufs);

	glBindVertexArray(m_quad_vao);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, d_depth_r);
	m_s_computeH->setUnif("zTex", 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, d_normal_D);
	m_s_computeH->setUnif("normalDTex", 1);
	
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}

void SSFRendererImpl::updateDepth() {
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glDisable(GL_BLEND);

	m_s_update_depth->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_update_depth->setUnif("p_n", i.n);
	m_s_update_depth->setUnif("p_f", i.f);
	m_s_update_depth->setUnif("s_w", (int)m_width);
	m_s_update_depth->setUnif("s_h", (int)m_height);
	m_s_update_depth->setUnif("zImg", 0);
	m_s_update_depth->setUnif("k", m_k);

	glBindImageTexture(0, d_depth_r, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	glBindTexture(GL_TEXTURE_2D, d_H);

	glDisable(GL_DEPTH_TEST);

	GLenum bufs[] = { GL_COLOR_ATTACHMENT2 /* d_H */ };
	glDrawBuffers(1, bufs);

	glBindVertexArray(m_quad_vao);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, d_depth_r);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, d_normal_D);

	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}