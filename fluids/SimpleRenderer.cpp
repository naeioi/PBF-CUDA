#include "SimpleRenderer.h"

#include <cstdlib>
#include <GLFW\glfw3.h>
#include <glm/common.hpp>

void SimpleRenderer::init() {

	m_width = WINDOW_WIDTH;
	m_height = WINDOW_HEIGHT;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(m_width, m_height, "Fluid", nullptr, nullptr);
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

	/* Resource allocation in constructor */
	glm::vec3 pos(2.f, -2.f, 2.f);
	float aspect = (float)WINDOW_WIDTH / WINDOW_HEIGHT;

	m_camera = new Camera(pos, aspect);
	/* This will loaded shader from shader/simple.cpp automatically */
	m_shader = new Shader();

	glGenVertexArrays(1, &d_vao);
	glGenVertexArrays(1, &d_bbox_vao);

	glGenBuffers(1, &d_bbox_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferData(GL_ARRAY_BUFFER, 12 * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindVertexArray(d_bbox_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}

void SimpleRenderer::__framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	m_width = width;
	m_height = height;
	glViewport(0, 0, width, height);
	m_camera->setAspect((float)width / height);
}

void SimpleRenderer::__binding() {
	// glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetWindowUserPointer(m_window, this);
	glfwSetWindowSizeCallback(m_window, [](GLFWwindow *win, int width, int height) {
		((SimpleRenderer*)(glfwGetWindowUserPointer(win)))->__framebuffer_size_callback(win, width, height);
	});
}

void SimpleRenderer::__render() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* == draw cubes == */
	if (m_shader->loaded()) {
		m_shader->use();

		m_camera->use(Shader::now());

		/* draw particles */
		glBindVertexArray(d_vao);
		glDrawArrays(GL_POINTS, 0, m_nparticle);
		glBindVertexArray(d_bbox_vao);
		glDrawArrays(GL_LINES, 0, 12 * 2);
	}

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

	/* == Bounding box == */
	float x1 = fmin(m_ulim.x, m_llim.x), x2 = fmax(m_ulim.x, m_llim.x),
		y1 = fmin(m_ulim.y, m_llim.y), y2 = fmax(m_ulim.y, m_llim.y),
		z1 = fmin(m_ulim.z, m_llim.z), z2 = fmax(m_ulim.z, m_llim.z);

	glm::vec3 lines[][2] = {
		{ glm::vec3(x1, y1, z1), glm::vec3(x2, y1, z1) },
		{ glm::vec3(x1, y1, z2), glm::vec3(x2, y1, z2) },
		{ glm::vec3(x1, y2, z1), glm::vec3(x2, y2, z1) },
		{ glm::vec3(x1, y2, z2), glm::vec3(x2, y2, z2) },

		{ glm::vec3(x1, y1, z1), glm::vec3(x1, y2, z1) },
		{ glm::vec3(x1, y1, z2), glm::vec3(x1, y2, z2) },
		{ glm::vec3(x2, y1, z1), glm::vec3(x2, y2, z1) },
		{ glm::vec3(x2, y1, z2), glm::vec3(x2, y2, z2) },

		{ glm::vec3(x1, y1, z1), glm::vec3(x1, y1, z2) },
		{ glm::vec3(x1, y2, z1), glm::vec3(x1, y2, z2) },
		{ glm::vec3(x2, y1, z1), glm::vec3(x2, y1, z2) },
		{ glm::vec3(x2, y2, z1), glm::vec3(x2, y2, z2) } };
	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lines), lines);

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
	else fexit(0);
}