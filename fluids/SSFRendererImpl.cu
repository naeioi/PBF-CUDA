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
	fprintf(stderr, "begin SSFRendererImpl()");

	/* TODO: consider how to handle resolution change */
	this->m_camera = camera;
	this->m_width = width;
	this->m_height = height;
	this->m_pi = camera->getProjectionInfo();

	fprintf(stderr, "flag1 SSFRendererImpl()\n");

	/* Allocate depth / position / normal texture */
	glGenTextures(1, &d_pos);
	glGenTextures(1, &d_depth);
	glGenTextures(1, &d_normal);

	fprintf(stderr, "flag2 SSFRendererImpl()\n");

	glBindTexture(GL_TEXTURE_2D, d_pos);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	/* TODO: check effect of GL_NEAREST */
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, d_normal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	fprintf(stderr, "flag3 SSFRendererImpl()\n");

	/* TODO: Bind texture to CUDA resource */
	/*checkCudaErrors(cudaGraphicsGLRegisterImage(&dc_pos, d_pos, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterImage(&dc_depth, d_pos, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterImage(&dc_normal, d_normal, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));*/

	/* Allocate & Binding depth texture to framebuffer */
	glGenFramebuffers(1, &d_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, d_depth, 0);

	/* Attach one color buffer, this is mandatory */
	uint colorTex;
	glGenTextures(1, &colorTex);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

	checkFramebufferComplete();
	checkGLErr();
	fprintf(stderr, "flag4 SSFRendererImpl()\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	/* Load shaders */
	m_s_get_depth = new Shader(Filename("SSFget_depth_vertex.glsl"), Filename("SSFget_depth_fragment.glsl"));
	fprintf(stderr, "break shader SSFRendererImpl()");
	m_s_put_depth = new Shader(Filename("SSFput_depth_vertex.glsl"), Filename("SSFput_depth_fragment.glsl"));

	fprintf(stderr, "flag5 SSFRendererImpl()\n");

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

	fprintf(stderr, "flag6 SSFRendererImpl()\n");
}

void SSFRendererImpl::destroy() {
	// if (!dc_depth) return;
	/* TODO */
}

void SSFRendererImpl::__render() {

	printf("__render()\n");

	/* Render to framebuffer */
	glBindFramebuffer(GL_FRAMEBUFFER, d_fbo);

	m_s_get_depth->use();
	m_camera->use(Shader::now());
	m_s_get_depth->setUnif("pointRadius", 50.f);

	glEnable(GL_DEPTH_TEST);
	glBindVertexArray(p_vao);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		fexit(-1, "Framebuffer not complete\n");

	glClear(GL_DEPTH_BUFFER_BIT);
	glDrawArrays(GL_POINTS, 0, m_nparticle);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	printf("Render to framebuffer\n");

	/* Draw depth in greyscale */
	m_s_put_depth->use();
	m_camera->use(Shader::now());
	glDisable(GL_DEPTH_TEST);
	glBindVertexArray(m_quad_vao);
	glBindTexture(GL_TEXTURE_2D, d_depth);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glEnable(GL_DEPTH_TEST);

	printf("Render quad\n");
}

void SSFRendererImpl::render(uint p_vao, int nparticle) {

	this->p_vao = p_vao;
	this->m_nparticle = nparticle;

	__render();
}