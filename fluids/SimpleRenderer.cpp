#include "shader/simple.cpp"
#include "SimpleRenderer.h"

#include <cstdlib>
#include <GLFW\glfw3.h>

SimpleRenderer::SimpleRenderer()
{
	/* Resource allocation in constructor */

	m_camera = new Camera();
	/* This will loaded shader from shader/simple.cpp automatically */
	m_shader = new Shader();

	/* State initialization in init() */
	init();
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

	__binding();
}

void SimpleRenderer::__binding() {
	/* currently none */
}

void SimpleRenderer::__render() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* == draw cubes == */
	if (m_shader->loaded()) {
		m_shader->use();

		m_camera->use(Shader::now());

		/* draw */
		glDrawArrays(GL_POINTS, d_pos, m_nparticle);
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

	__render();
}