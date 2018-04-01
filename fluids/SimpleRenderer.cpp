#include "SimpleRenderer.h"

#include <cstdlib>
#include <GLFW\glfw3.h>
#include <glm/common.hpp>

SimpleRenderer::SimpleRenderer()
{
	/* State initialization in init() */
	init();

	/* Resource allocation in constructor */
	glm::vec3 pos(1.f, -1.f, 1.f);
	float aspect = (float) WINDOW_WIDTH / WINDOW_HEIGHT;

	m_camera = new Camera(pos, aspect);
	/* This will loaded shader from shader/simple.cpp automatically */
	m_shader = new Shader();

	glGenVertexArrays(1, &d_vao);
}

void SimpleRenderer::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Fluid", nullptr, nullptr);
	if (m_window == nullptr) {
		printf("Failed to create GLFW window\n");
		glfwTerminate();
		fexit(-1);
	}
	glfwMakeContextCurrent(m_window);

	if (!gladLoadGL()) {
		printf("Failed to initialize GLAD\n");
		fexit(-1);
	}

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glEnable(GL_DEPTH_TEST);

	__binding();
}

void SimpleRenderer::__binding() {
	glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void SimpleRenderer::__render() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* == draw cubes == */
	if (m_shader->loaded()) {
		m_shader->use();

		m_camera->use(Shader::now());

		/* draw */
		glDrawArrays(GL_POINTS, 0, m_nparticle);
	}

	/* == draw bounding box == */
	/* TODO */
}

SimpleRenderer::~SimpleRenderer()
{
	if (m_camera) delete m_camera;
	if (m_shader) delete m_shader;
}

void SimpleRenderer::render(uint pos, int nparticle) {
	/** 
	 * @input pos vertex buffer object 
	 */
	d_pos = pos;
	m_nparticle = nparticle;

	glBindVertexArray(d_vao);
	glBindBuffer(GL_ARRAY_BUFFER, d_pos);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	if (!glfwWindowShouldClose(m_window)) {
		// processInput(m_window);
		__render();
		glfwSwapBuffers(m_window);
		// ctx.syncService.newFrame();
		/*if (ctx.lockFPS > 0) {
			float sleepTime = 1000.f * (1.f / ctx.lockFPS - ctx.syncService.frameTime());
			if (sleepTime < 0) sleepTime = 0;
			Sleep(sleepTime);
		}*/
		glfwPollEvents();
	}
}