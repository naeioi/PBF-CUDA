#include "SimpleRenderer.h"
#include "Input.h"

#include <cstdlib>
#include <GLFW\glfw3.h>
#include <glm/common.hpp>
#include <glm/gtx/rotate_vector.hpp>

void SimpleRenderer::init() {

	m_width = WINDOW_WIDTH;
	m_height = WINDOW_HEIGHT;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_input = new Input();
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

void SimpleRenderer::__window_size_callback(GLFWwindow* window, int width, int height) {
	m_width = width;
	m_height = height;
	glViewport(0, 0, width, height);
	m_camera->setAspect((float)width / height);
}

void SimpleRenderer::__mouse_button_callback(GLFWwindow *w, int button, int action, int mods) {
	Input::Pressed updown = action == GLFW_PRESS ? Input::DOWN : Input::UP;
	if (button == GLFW_MOUSE_BUTTON_LEFT)
		m_input->left_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
		m_input->right_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
		m_input->mid_mouse = updown;
}

void SimpleRenderer::__mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
	m_input->updateMousePos(glm::vec2(xpos, ypos));

	/* -- Camera control -- */

	/* Rotating */
	if (m_input->left_mouse == Input::DOWN) {
		glm::vec2 scr_d = m_input->getMouseDiff();
		glm::vec3 pos = m_camera->getPos(), front = m_camera->getFront(), center = pos + front, up = m_camera->getUp();
		glm::vec3 axis;
		/* For horizontal panning, rotate camera within plane perpendicular to `up' direction */
		if (scr_d.x != 0) {
			axis = glm::normalize(m_camera->getUp());
			/* for now, manually update pos, front and up in renderer */
			front = glm::rotate(front, scr_d.x * Input::SCREEN_ROTATE_RATE, axis);
			pos = center - front;

			m_camera->setPos(pos);
			m_camera->setFront(front);
		}
		/* For verticle panning, rotate camera within plane perpendicular to cross(up, front) direction */
		if (scr_d.y != 0) {
			axis = glm::normalize(glm::cross(m_camera->getUp(), front));

			front = glm::rotate(front, scr_d.y * Input::SCREEN_ROTATE_RATE, axis);
			up = glm::rotate(up, scr_d.y * Input::SCREEN_ROTATE_RATE, axis);
			pos = center - front;

			m_camera->setPos(pos);
			m_camera->setUp(up);
			m_camera->setFront(front);
		}
	}

	/* Panning */
	if (m_input->right_mouse == Input::DOWN) {
		glm::vec3 pos = m_camera->getPos(), front = m_camera->getFront(), up = m_camera->getUp();
		glm::vec3 dx, dy;
		glm::vec2 scr_d = m_input->getMouseDiff();

		dx = glm::normalize(glm::cross(front, up));
		dy = glm::normalize(up);

		pos += scr_d.x * dx + scr_d.y * dy;
		m_camera->setPos(pos);
	}
}

void SimpleRenderer::__mouse_scroll_callback(GLFWwindow *w, float dx, float dy) {
	const float min_d = 0.1f, max_d = 10.f;
	glm::vec3 d = m_camera->getFront(), pos = m_camera->getPos();
	if (dy > 0) {
		if (d.length() < min_d) return;
		pos += d * Input::SCREEN_SCROLL_RATE;
		m_camera->setPos(pos);
	}
	else {
		if (d.length() > max_d) return;
		pos -= d * Input::SCREEN_SCROLL_RATE;
		m_camera->setPos(pos);
	}
}

void SimpleRenderer::__binding() {
	// glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	/* glfw input callback requires static function (not class method) 
	 * as a workaround, glfw provides a 'user pointer', 
	 * by which static function gets access to class method
	 */
	glfwSetWindowUserPointer(m_window, this);

	/* Windows resize */
	glfwSetWindowSizeCallback(m_window, [](GLFWwindow *win, int width, int height) {
		((SimpleRenderer*)(glfwGetWindowUserPointer(win)))->__window_size_callback(win, width, height);
	});

	/* Mouse move */
	glfwSetCursorPosCallback(m_window, [](GLFWwindow *w, double xpos, double ypos) {
		((SimpleRenderer*)(glfwGetWindowUserPointer(w)))->__mouse_move_callback(w, xpos, ypos);
	});

	/* Mouse Button */
	glfwSetMouseButtonCallback(m_window, [](GLFWwindow* w, int button, int action, int mods) {
		((SimpleRenderer*)(glfwGetWindowUserPointer(w)))->__mouse_button_callback(w, button, action, mods);
	});

	/* Mouse Scroll */
	glfwSetScrollCallback(m_window, [](GLFWwindow *w, double dx, double dy) {
		((SimpleRenderer*)(glfwGetWindowUserPointer(w)))->__mouse_scroll_callback(w, dx, dy);
	});
}

void SimpleRenderer::__processInput() {

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
	/* TODO: m_window, input */
	if (m_input) delete m_input;
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
		__processInput();
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