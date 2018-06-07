#include "helper.h"
#include "SSFRendererImpl.h"
#include "Logger.h"
#include <GLFW\glfw3.h>
#include <glad\glad.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include "GUIParams.h"

static float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
    // positions   // texCoords
    -1.0f,  1.0f,  0.0f, 1.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
    1.0f, -1.0f,  1.0f, 0.0f,

    -1.0f,  1.0f,  0.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 0.0f,
    1.0f,  1.0f,  1.0f, 1.0f
};

SSFRendererImpl::SSFRendererImpl(Camera *camera, int width, int height, uint sky_texture)
{
	loadParams();
	{
		float n1 = 1.3333f;
		float t = (n1 - 1) / (n1 + 1);
		m_r0 = t * t;
	}
	m_ab = 0;

	/* TODO: consider how to handle resolution change */
	this->m_camera = camera;
	this->m_width = width;
	this->m_height = height;
	this->m_pi = camera->getProjectionInfo();
	this->d_sky = sky_texture;

	/* Allocate depth / normal_D / H texture */
	glGenTextures(1, &d_depth);
	glGenTextures(1, &d_depth_a);
	glGenTextures(1, &d_depth_b);
	glGenTextures(1, &d_normal_D);
	glGenTextures(1, &d_H);
	glGenTextures(1, &d_thick);

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
	glBindTexture(GL_TEXTURE_2D, d_depth_a);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_depth_b);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_H);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLErr();
	glBindTexture(GL_TEXTURE_2D, d_thick);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	checkGLErr();

	/* TODO: Bind texture to CUDA resource */
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_normal_D, d_normal_D, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	/* CUDA does not support interop with GL_DEPTH_COMPONENT texture ! 
	 * As a workaround, first render to a depth texture (d_depth)
	 * then copy depth texture to a color texture (d_depth_a), which contains only red channel 
	 */
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_depth, d_depth_a, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	//checkCudaErrors(cudaGraphicsGLRegisterImage(&dcr_H, d_H, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	/* Allocate framebuffer & Binding depth texture */
	glGenFramebuffers(1, &d_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, d_depth, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, d_depth_a, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, d_depth_b, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, d_normal_D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, d_H, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, d_thick, 0);

	checkFramebufferComplete();
	checkGLErr();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Load shaders */
	m_s_get_depth = new Shader(Filename("SSFget_depth.v.glsl"), Filename("SSFget_depth.f.glsl"));
	m_s_get_thick = new Shader(Filename("SSFget_thick.v.glsl"), Filename("SSFget_thick.f.glsl"));
	m_s_shading = new Shader(Filename("SSFshading.v.glsl"), Filename("SSFshading.f.glsl"));
	m_s_restore_normal = new Shader(Filename("SSFrestore_normal.v.glsl"), Filename("SSFrestore_normal.f.glsl"));
	m_s_computeH = new Shader(Filename("SSFcomputeH.v.glsl"), Filename("SSFcomputeH.f.glsl"));
	m_s_update_depth = new Shader(Filename("SSFupdate_depth.v.glsl"), Filename("SSFupdate_depth.f.glsl"));
	m_s_smooth_depth = new Shader(Filename("SSFsmooth_depth.v.glsl"), Filename("SSFsmooth_depth.f.glsl"));

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

void SSFRendererImpl::render(uint p_vao, int nparticle) {

	auto &logger = Logger::getInstance();

	this->p_vao = p_vao;
	this->m_nparticle = nparticle;
	loadParams();
	m_ab = 0;

	logger.logTime(Logger::DEPTH_START);
	renderDepth();
	logger.logTime(Logger::DEPTH_END);

	logger.logTime(Logger::THICK_START);
	renderThick();
	logger.logTime(Logger::THICK_END);

	// Algo 1. Compute H and update depth
	/*for (int i = 0; i < m_niter; i++) {
	restoreNormal();
	computeH();
	updateDepth();
	}*/

	logger.logTime(Logger::SMOOTH_START);
	// Algo 2. Smooth filtering
	for (int i = 0; i < m_niter; i++) {
		/* Flip pingpong flag BEFORE each step */
		m_ab = !m_ab;
		smoothDepth();
	}
	logger.logTime(Logger::SMOOTH_END);
	
	logger.logTime(Logger::NORMAL_START);
	restoreNormal();
	logger.logTime(Logger::NORMAL_END);

	logger.logTime(Logger::SHADING_START);
	shading();
	logger.logTime(Logger::SHADING_END);
}

void SSFRendererImpl::renderDepth() {
	/* After renderDepth(), z_c is store at d_depth 
	 * Linearize depth (z_e) is stored at d_depth_a  
	 */

	/* Render to framebuffer */
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	// glEnable(GL_BLEND);
	glDisable(GL_BLEND);

	/* Reset depth_r to maximum */
	GLfloat inf[] = { 100.f }, zero[] = { 0.f };
	glClearTexImage(d_depth_a, 0, GL_RED, GL_FLOAT, inf);
	checkGLErr();
	glClearTexImage(d_depth_b, 0, GL_RED, GL_FLOAT, inf);
	checkGLErr();

	/* Have to assign COLOR_ATTACHMENT0 to first drawbuffer
	 * because later we assign COLOR_ATTACHMENT2 to first drawbuffer
	 */
	GLenum bufs[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, bufs);

	m_s_get_depth->use();
	m_camera->use(Shader::now());

	/* TODO: encapsulate uniforms into */
	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_get_depth->setUnif("s_h", m_height);
	m_s_get_depth->setUnif("p_t", i.t);
	m_s_get_depth->setUnif("p_n", i.n);
	m_s_get_depth->setUnif("p_f", i.f);
	m_s_get_depth->setUnif("r", 0.1f * 0.5f);
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

void SSFRendererImpl::renderThick() {
	/* After renderDepth(), z_c is store at d_depth
	* Linearize depth (z_e) is stored at d_depth_a
	*/

	/* Render to framebuffer */
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glEnable(GL_BLEND);

	/* Reset depth_r to maximum */
	GLfloat zero[] = { 0.f };
	glClearTexImage(d_thick, 0, GL_RED, GL_FLOAT, zero);
	checkGLErr();

	/* Have to assign COLOR_ATTACHMENT0 to first drawbuffer
	* because later we assign COLOR_ATTACHMENT2 to first drawbuffer
	*/
	GLenum bufs[] = { GL_COLOR_ATTACHMENT4 /* d_thick */ };
	glDrawBuffers(1, bufs);

	/* Disable blend for depth & Set additive blend for thickness */
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	glBlendFuncSeparateiARB(0, GL_ONE, GL_ONE, GL_ONE, GL_ONE);

	m_s_get_thick->use();
	m_camera->use(Shader::now());

	/* TODO: encapsulate uniforms into */
	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_get_thick->setUnif("s_h", m_height);
	m_s_get_thick->setUnif("p_t", i.t);
	m_s_get_thick->setUnif("p_n", i.n);
	m_s_get_thick->setUnif("p_f", i.f);
	m_s_get_thick->setUnif("r", 0.1f * 0.5f);
	m_s_get_thick->setUnif("pointRadius", 50.f);

	glDisable(GL_DEPTH_TEST);
	glBindVertexArray(p_vao);

	glDrawArrays(GL_POINTS, 0, m_nparticle);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void SSFRendererImpl::shading() {

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Draw depth in greyscale */
	m_s_shading->use();
	m_camera->use(Shader::now());
	m_s_shading->setUnif("iview", m_camera->getInverseView());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_shading->setUnif("p_n", i.n);
	m_s_shading->setUnif("p_f", i.f);
	m_s_shading->setUnif("p_t", i.t);
	m_s_shading->setUnif("p_r", i.r);
	m_s_shading->setUnif("r0", m_r0);

	m_s_shading->setUnif("shading_option", GUIParams::getInstance().shading_option);

	glEnable(GL_DEPTH_TEST);
	glBindVertexArray(m_quad_vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, zTex2());
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, d_normal_D);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, d_thick);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_CUBE_MAP, d_sky);

	glBlendFuncSeparateiARB(0, GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);

	m_s_shading->setUnif("zTex", 0);
	m_s_shading->setUnif("normalDTex", 1);
	m_s_shading->setUnif("thickTex", 2);
	m_s_shading->setUnif("skyTex", 3);

	// glClear(GL_COLOR_BUFFER_BIT);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void SSFRendererImpl::loadParams()
{
	const GUIParams &params = GUIParams::getInstance();

	m_niter = params.smooth_niter;
	m_kernel_r = params.kernel_r;
	m_blur_r = 1 / params.sigma_r;
	m_blur_z = 1 / params.sigma_z;
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

	m_s_restore_normal->setUnif("keep_edge", GUIParams::getInstance().keep_edge);

	glDisable(GL_DEPTH_TEST);

	GLfloat black[] = { 0.f, 0.f, 0.f, 0.f };
	GLenum bufs[] = { GL_COLOR_ATTACHMENT2 /* d_normal_D */ };
	glDrawBuffers(1, bufs);
	glClearTexImage(d_normal_D, 0, GL_RGBA, GL_FLOAT, black);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, zTex2());
	m_s_restore_normal->setUnif("zTex", 0);

	glBindVertexArray(m_quad_vao);
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
	glBindTexture(GL_TEXTURE_2D, d_depth_a);
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

	glBindImageTexture(0, d_depth_a, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, d_H);

	m_s_update_depth->setUnif("zImg", 0);
	m_s_update_depth->setUnif("hTex", 1);

	glDisable(GL_DEPTH_TEST);

	glBindVertexArray(m_quad_vao);

	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}

void SSFRendererImpl::smoothDepth()
{
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glDisable(GL_BLEND);

	m_s_smooth_depth->use();
	m_camera->use(Shader::now());

	ProjectionInfo i = m_camera->getProjectionInfo();
	m_s_smooth_depth->setUnif("p_n", i.n);
	m_s_smooth_depth->setUnif("p_f", i.f);
	m_s_smooth_depth->setUnif("s_w", (int)m_width);
	m_s_smooth_depth->setUnif("s_h", (int)m_height);

	m_s_smooth_depth->setUnif("blur_option", GUIParams::getInstance().blur_option);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, zTex1());
	glActiveTexture(GL_TEXTURE1);
	glBindImageTexture(1, zTex2(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

	m_s_smooth_depth->setUnif("zA", 0);
	m_s_smooth_depth->setUnif("zB", 1);

	m_s_smooth_depth->setUnif("kernel_r", m_kernel_r);
	m_s_smooth_depth->setUnif("blur_r", m_blur_r);
	m_s_smooth_depth->setUnif("blur_z", m_blur_z);

	glDisable(GL_DEPTH_TEST);
	glBindVertexArray(m_quad_vao);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
}
