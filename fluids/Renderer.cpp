#include "Renderer.h"
#include "Input.h"
#include "Logger.h"

#include <cstdlib>

#include <GLFW\glfw3.h>
#include <nanogui\nanogui.h>
#include <glm/common.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <stb.h>

extern float SKYBOX_VERTICES[];
extern float GROUND_VERTICES[];
static uint loadCubemap(char **faces);

void Renderer::init(const glm::vec3 &cam_pos, const glm::vec3 &cam_focus) {

	m_draw_sky = true;
	m_draw_fluid = true;
	m_draw_bbox = true;
	m_draw_ground = true;

	m_width = WINDOW_WIDTH;
	m_height = WINDOW_HEIGHT;

	glfwInit();
	glfwSetTime(0);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_SAMPLES, 0);
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);
	glfwWindowHint(GLFW_STENCIL_BITS, 8);
	glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

	m_input = &Input::getInstance();

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

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		throw std::runtime_error("Could not initialize GLAD!");
	glGetError();

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	// glEnable(GL_BLEND);
	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	/* Init nanogui */
	// m_gui_screen = new nanogui::Screen(Eigen::Vector2i(1024, 768), "Fluids");
	m_gui_screen = new nanogui::Screen();
	m_gui_screen->initialize(m_window, true);
	m_gui_screen->setSize(Eigen::Vector2i(800, 600));

	int width_, height_;
	glfwGetFramebufferSize(m_window, &width_, &height_);
	m_width = width_; m_height = height_;
	glViewport(0, 0, width_, height_);
	glfwSwapInterval(0);
	glfwSwapBuffers(m_window);

	m_gui_form = new nanogui::FormHelper(m_gui_screen);
	m_gui_win = m_gui_form->addWindow(Eigen::Vector2i(30, 30), "Parameters");

	GUIParams &params = GUIParams::getInstance();
	m_gui_form->addVariable("# Frame", m_input->frameCount)->setEditable(false);
	m_gui_form->addVariable("# Iter", params.niter)->setSpinnable(true);
	// m_gui_form->addVariable("pho0", params.pho0)->setSpinnable(true);
	// m_gui_form->addVariable("g", params.g)->setSpinnable(true);
	// m_gui_form->addVariable("h", params.h)->setSpinnable(true);
	// m_gui_form->addVariable("dt", params.dt)->setSpinnable(true);
	// m_gui_form->addVariable("lambda_eps", params.lambda_eps)->setSpinnable(true);
	// m_gui_form->addVariable("delta_q", params.delta_q)->setSpinnable(true);
	// m_gui_form->addVariable("k_corr", params.k_corr)->setSpinnable(true);
	// m_gui_form->addVariable("n_corr", params.n_corr)->setSpinnable(true);
	// m_gui_form->addVariable("k_boundary", params.k_boundaryDensity)->setSpinnable(true);
	// m_gui_form->addVariable("c_XSPH", params.c_XSPH)->setSpinnable(true);
	// m_gui_form->addVariable("Highlight #", m_input->hlIndex)->setSpinnable(true);
	
	auto smooth_niter = m_gui_form->addVariable("Smooth # Iter", params.smooth_niter); 
	smooth_niter->setMinMaxValues(0, 60);
	smooth_niter->setSpinnable(true);

	auto kernel_r = m_gui_form->addVariable("kernel_r", params.kernel_r);
	kernel_r->setMinMaxValues(0, 20);
	kernel_r->setSpinnable(true);

	auto sigma_r = m_gui_form->addVariable("sigma_r", params.sigma_r);
	sigma_r->setMinMaxValues(0.f, 10.f);
	sigma_r->setSpinnable(true);

	auto sigma_z = m_gui_form->addVariable("sigma_z", params.sigma_z);
	sigma_z->setMinMaxValues(0.f, 1.f);
	sigma_z->setSpinnable(true);

	m_gui_form->addVariable("shading_option", params.shading_option)->setSpinnable(true);
	m_gui_form->addVariable("keep_edge", params.keep_edge)->setSpinnable(true);
	m_gui_form->addVariable("blur_option", params.blur_option)->setSpinnable(true);

	m_gui_form->addButton("Next Frame", [this]() { m_nextFrameBtnCb();  });
	auto runBtn = m_gui_form->addButton("Run", []() {});
	runBtn->setFlags(nanogui::Button::ToggleButton);
	runBtn->setChangeCallback([this](bool state) { m_input->running = state; });
	
	auto lastFrameBtn = m_gui_form->addButton("Last Frame", []() {});
	lastFrameBtn->setFlags(nanogui::Button::ToggleButton);
	lastFrameBtn->setChangeCallback([this](bool state) { m_input->lastFrame = state; });

	auto movingBtn = m_gui_form->addButton("Sweep Boundary", []() {});
	movingBtn->setFlags(nanogui::Button::ToggleButton);
	movingBtn->setChangeCallback([this](bool state) {
		printf("moving Btn=%d\n", state);
		auto &input = Input::getInstance();
		if (state) {
			input.moving = true;
			input.startMovingFrame = input.frameCount;
		}
		else {
			input.moving = false;
		}
	});

	auto logTime = m_gui_form->addButton("Log Time", []() {});
	logTime->setFlags(nanogui::Button::ToggleButton);
	logTime->setChangeCallback([this](bool state) { Logger::getInstance().toggleLogTime(state); });

	m_gui_form->addButton("Report Time", []() { Logger::getInstance().report();  });

	m_gui_screen->setVisible(true);
	m_gui_screen->performLayout();
	// nanoWin->center();

	__binding();

	/* Resource allocation in constructor */
	float aspect = (float)WINDOW_WIDTH / WINDOW_HEIGHT;

	m_camera = new Camera(cam_pos, cam_focus, aspect);
	/* This will loaded shader from shader/simple.cpp automatically */
	m_box_shader = new Shader(Path("shader/box.v.glsl"), Path("shader/box.f.glsl"));
	m_particle_shader = new Shader(Path("shader/particle.v.glsl"), Path("shader/particle.f.glsl"));

	/* Load skybox */
	char *sky_faces[] = { 
		"skybox/right.jpg",		/* +x */
		"skybox/left.jpg",		/* -x */
		"skybox/front.jpg",		/* +y */
		"skybox/back.jpg",		/* -y */
		"skybox/top.jpg",		/* +z */
		"skybox/bottom.jpg"		/* -z */
	};
	d_sky_texture = loadCubemap(sky_faces);
	m_sky_shader = new Shader(Path("shader/sky.v.glsl"), Path("shader/sky.f.glsl"));

	/* Ground shader */
	m_ground_shader = new Shader(Path("shader/ground.v.glsl"), Path("shader/ground.f.glsl"));

	/* Allow space for d_vao, d_bbox_vao, d_sky_vao */
	glGenVertexArrays(1, &d_vao);
	glGenVertexArrays(1, &d_bbox_vao);
	glGenVertexArrays(1, &d_sky_vao);
	glGenVertexArrays(1, &d_ground_vao);
	
	/* Bind bbox vbo to vao */
	glGenBuffers(1, &d_bbox_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_bbox_vbo);
	glBufferData(GL_ARRAY_BUFFER, 12 * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindVertexArray(d_bbox_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	/* Bind sky vbo to vao */
	glGenBuffers(1, &d_sky_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_sky_vbo);
	glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 3 * 3 * sizeof(float), SKYBOX_VERTICES, GL_STATIC_DRAW);
	glBindVertexArray(d_sky_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	/* Bind ground vbo to vao */
	glGenBuffers(1, &d_ground_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, d_ground_vbo);
	glBufferData(GL_ARRAY_BUFFER, 6 * 5 * sizeof(float), GROUND_VERTICES, GL_STATIC_DRAW);
	glBindVertexArray(d_ground_vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	/* SSFRenderer */
	m_SSFrenderer = new SSFRenderer(m_camera, width_, height_, d_sky_texture);

}

void Renderer::__window_size_callback(GLFWwindow* window, int width, int height) {
	m_width = width;
	m_height = height;
	glViewport(0, 0, width, height);
	m_camera->setAspect((float)width / height);
	m_gui_screen->resizeCallbackEvent(width, height);
}

void Renderer::__mouse_button_callback(GLFWwindow *w, int button, int action, int mods) {
	if (m_gui_screen->mouseButtonCallbackEvent(button, action, mods)) return;

	Input::Pressed updown = action == GLFW_PRESS ? Input::DOWN : Input::UP;
	if (button == GLFW_MOUSE_BUTTON_LEFT)
		m_input->left_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
		m_input->right_mouse = updown;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
		m_input->mid_mouse = updown;
}

void Renderer::__mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
	if (m_gui_screen->cursorPosCallbackEvent(xpos, ypos)) return;

	m_input->updateMousePos(glm::vec2(xpos, ypos));

	/* -- Camera control -- */

	/* Rotating */
	glm::vec2 scr_d = m_input->getMouseDiff();
	glm::vec3 pos = m_camera->getPos(), front = m_camera->getFront(), center = pos + front, up = m_camera->getUp();
	glm::vec3 cam_d = scr_d.x * -glm::normalize(glm::cross(front, up)) + scr_d.y * glm::normalize(up);

	if (m_input->left_mouse == Input::DOWN)
		m_camera->rotate(scr_d);

	/* Panning */
	if (m_input->right_mouse == Input::DOWN)
		m_camera->pan(scr_d);
	
}

void Renderer::__key_callback(GLFWwindow *w, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_V && action == GLFW_RELEASE) {
		m_gui_win->setVisible(!m_gui_win->visible());
	}
	else 
		m_gui_screen->keyCallbackEvent(key, scancode, action, mods);
}

void Renderer::__mouse_scroll_callback(GLFWwindow *w, float dx, float dy) {
	if(m_gui_screen->scrollCallbackEvent(dx, dy)) return;
	m_camera->zoom(dy);
}

void Renderer::__char_callback(GLFWwindow *w, unsigned int codepoint) {
	m_gui_screen->charCallbackEvent(codepoint);
}

void Renderer::__binding() {
	// glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	/* glfw input callback requires static function (not class method) 
	 * as a workaround, glfw provides a 'user pointer', 
	 * by which static function gets access to class method
	 */
	glfwSetWindowUserPointer(m_window, this);

	/* Windows resize */
	glfwSetWindowSizeCallback(m_window, [](GLFWwindow *win, int width, int height) {
		((Renderer*)(glfwGetWindowUserPointer(win)))->__window_size_callback(win, width, height);
	});

	/* Mouse move */
	glfwSetCursorPosCallback(m_window, [](GLFWwindow *w, double xpos, double ypos) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_move_callback(w, xpos, ypos);
	});

	/* Mouse Button */
	glfwSetMouseButtonCallback(m_window, [](GLFWwindow* w, int button, int action, int mods) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_button_callback(w, button, action, mods);
	});

	/* Mouse Scroll */
	glfwSetScrollCallback(m_window, [](GLFWwindow *w, double dx, double dy) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__mouse_scroll_callback(w, dx, dy);
	});

	/* GUI keyboard input */
	glfwSetKeyCallback(m_window,
		[](GLFWwindow *w, int key, int scancode, int action, int mods) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__key_callback(w, key, scancode, action, mods);
	});

	glfwSetCharCallback(m_window,
		[](GLFWwindow *w, unsigned int codepoint) {
		((Renderer*)(glfwGetWindowUserPointer(w)))->__char_callback(w, codepoint);
	});
}

bool move = false;
void Renderer::__processInput() {
	if (glfwGetKey(m_window, GLFW_KEY_M) == GLFW_PRESS)
		move = true;
}

void Renderer::__render() {
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_draw_sky) {
		/* Sky */
		glDepthMask(GL_FALSE);
		m_sky_shader->use();
		m_camera->use(Shader::now(), true);

		glBindVertexArray(d_sky_vao);
		glBindTexture(GL_TEXTURE_CUBE_MAP, d_sky_texture);
		glDrawArrays(GL_TRIANGLES, 0, 36);

		glDepthMask(GL_TRUE);
	}

	if (m_draw_ground) {
		/* Ground */
		m_ground_shader->use();
		m_camera->use(Shader::now());

		glBindVertexArray(d_ground_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}

	if (!m_draw_fluid && m_particle_shader->loaded()) {
		/* Particle */
		m_particle_shader->use();
		m_camera->use(Shader::now());

		m_particle_shader->setUnif("color", glm::vec4(1.f, 0.f, 0.f, .1f));
		m_particle_shader->setUnif("pointRadius", GUIParams::getInstance().h);
		m_particle_shader->setUnif("pointScale", 500.f);
		m_particle_shader->setUnif("hlIndex", m_input->hlIndex);
		glBindVertexArray(d_vao);
		glDrawArrays(GL_POINTS, 0, m_nparticle);
	}
	else if (m_draw_fluid) {
		/* Fluid */
		m_SSFrenderer->render(d_vao, m_nparticle);
	}

	if (m_draw_bbox && m_box_shader->loaded()) {
		/* Bounding box */
		m_box_shader->use();
		m_camera->use(Shader::now());

		/* draw particles */
		glBindVertexArray(d_bbox_vao);
		m_box_shader->setUnif("color", glm::vec4(1.f, 1.f, 1.f, 1.f));
		glDrawArrays(GL_LINES, 0, 12 * 2);
	}


	/* Send to ffmpeg */
	// glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_buffer);
	// fwrite(m_buffer, sizeof(int)*m_width*m_height, 1, m_ffmpeg);
}

Renderer::~Renderer()
{
	if (m_camera) delete m_camera;
	if (m_box_shader) delete m_box_shader;
	if (m_particle_shader) delete m_particle_shader;
	/* TODO: m_window, input */
	if (m_input) delete m_input;
	// _pclose(m_ffmpeg);
}

void Renderer::render(uint pos, uint iid, int nparticle) {
	/** 
	 * @input pos vertex buffer object 
	 */
	d_iid = iid;
	d_pos = pos;
	m_nparticle = nparticle;

	glBindVertexArray(d_vao);
	glBindBuffer(GL_ARRAY_BUFFER, d_pos);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, d_iid);
	/* MUST use glVertexAttribIPointer, not glVertexAttribPointer, for uint attribute */
	glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, 0, (void*)0);
	glEnableVertexAttribArray(1);

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

	m_gui_form->refresh();

	if (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
		__processInput();
		__render();
		m_gui_screen->drawContents();
		m_gui_screen->drawWidgets();
		glfwSwapBuffers(m_window);
	}
	else fexit(0);
}

static uint loadCubemap(char **faces) {
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

	int width, height, nrChannels;
	for (unsigned int i = 0; i < 6; i++)
	{
		unsigned char *data = stbi_load(faces[i], &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
			);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return textureID;
}

void Renderer::setLim(const float3 & ulim, const float3 & llim)
{
	m_llim = llim;
	m_ulim = ulim;
}
