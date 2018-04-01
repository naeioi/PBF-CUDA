#pragma once
#include "Renderer.h"
#include "Camera.h"
#include "Shader.h"

#include <GLFW\glfw3.h>

class SimpleRenderer :
	public Renderer
{
public:
	SimpleRenderer();
	~SimpleRenderer();

	void render(uint pos, int m_nparticle);
private:

	void init();
	void __binding();
	void loop();
	void __render();

	int m_nparticle;
	/* Nsight debugging cannot work without a vao */
	uint d_vao;
	/* particle position vbo */
	uint d_pos;
	/* bounding box */
	float3 m_llim, m_ulim;

	/* Renderer states */
	Camera *m_camera;

	Shader *m_shader;

	GLFWwindow *m_window;
};

