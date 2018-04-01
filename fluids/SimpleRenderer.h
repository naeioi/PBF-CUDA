#pragma once
#include "Renderer.h"
#include "Camera.h"
#include "Shader.h"

#include <GLFW\glfw3.h>

class SimpleRenderer :
	public Renderer
{
public:
	SimpleRenderer(float3 ulim, float3 llim) : m_ulim(ulim), m_llim(llim) { init();  };
	~SimpleRenderer();

	void render(uint pos, int m_nparticle);
private:

	void init();
	void __binding();
	void loop();
	void __render();

	/* event callback */
	void __framebuffer_size_callback(GLFWwindow* window, int width, int height);

	int m_width, m_height;
	int m_nparticle;
	/* Nsight debugging cannot work without a vao */
	uint d_vao, d_bbox_vao, d_bbox_vbo;
	/* particle position vbo */
	uint d_pos;
	/* bounding box */
	float3 m_llim, m_ulim;

	/* Renderer states */
	Camera *m_camera;

	Shader *m_shader;

	GLFWwindow *m_window;
};

