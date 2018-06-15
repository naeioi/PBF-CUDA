#pragma once
#include "Renderer.h"
#include "Camera.h"
#include "Shader.h"
#include "Input.h"
#include "SSFRenderer.h"

#include "GUIParams.h"
#include <GLFW\glfw3.h>
#include <nanogui\nanogui.h>
#include <functional>

class SimpleRenderer :
	public Renderer
{
public:
	SimpleRenderer(
		const glm::vec3 &cam_pos, 
		const glm::vec3 &cam_focus, 
		float3 ulim, float3 llim, 
		std::function<void()> nextCb) 
		: m_ulim(ulim), m_llim(llim), m_nextFrameBtnCb(nextCb) { init(cam_pos, cam_focus); };
	~SimpleRenderer();

	void render(uint pos, uint iid, int m_nparticle);
	void setLim(const float3 &ulim, const float3 &llim);

	Input *m_input;

private:

	void init(const glm::vec3 &cam_pos, const glm::vec3 &cam_focus);
	void __binding();
	void __render();
	void __processInput();

	/* event callback */
	void __window_size_callback(GLFWwindow* window, int width, int height);
	void __mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
	void __mouse_button_callback(GLFWwindow* w, int button, int action, int mods);
	void __mouse_scroll_callback(GLFWwindow* w, float dx, float dy);
	void __key_callback(GLFWwindow *w, int key, int scancode, int action, int mods);
	void __char_callback(GLFWwindow *w, unsigned int codepoint);

	int m_width, m_height;
	int m_nparticle;
	/* Nsight debugging cannot work without a vao */
	uint d_vao, d_bbox_vao, d_bbox_vbo;
	/* particle position vbo */
	uint d_pos, d_iid;
	/* bounding box */
	float3 m_llim, m_ulim;

	SSFRenderer *m_SSFrenderer;

	/* Renderer states */
	Camera *m_camera;

	Shader *m_box_shader;
	Shader *m_particle_shader;

	GLFWwindow *m_window;

	/* NanoGUI */
	nanogui::Screen *m_gui_screen;
	nanogui::FormHelper *m_gui_form;
	nanogui::Window *m_gui_win;
	std::function<void()> m_nextFrameBtnCb;
	double m_dvar = 1.23456;

	int frameCount = 0;

	/* Cubemap */
	uint d_sky_texture;
	Shader *m_sky_shader;
	uint d_sky_vao, d_sky_vbo;

	/* Ground */
	uint d_ground_vao, d_ground_vbo;
	Shader *m_ground_shader;

	/* Rendering options */
	bool m_draw_sky;
	/* If false, draw particle as sphere instead */
	bool m_draw_fluid;
	bool m_draw_bbox;
	bool m_draw_ground;

	/* FFMPEG */
	int* m_buffer;
	FILE *m_ffmpeg;
};

