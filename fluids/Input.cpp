#include "Input.h"
#include <GLFW\glfw3.h>

const float Input::SCREEN_ROTATE_RATE = 0.005f;
const float Input::SCREEN_PAN_RATE = 0.002f;
const float Input::SCREEN_SCROLL_RATE = 0.1f;

Input::Input() {
	reset();
}

glm::vec2 Input::updateMousePos(glm::vec2 new_mouse)
{
	if (!last_mouse_valid) {
		last_mouse_valid = true;
		last_mouse = mouse = new_mouse;
	}
	else {
		last_mouse = mouse;
		mouse = new_mouse;
	}

	return mouse - last_mouse;
}

glm::vec2 Input::getMouseDiff() {
	return mouse - last_mouse;
}

void Input::reset() {
	last_mouse_valid = false;
	running = false;
	right_mouse = left_mouse = UP;
}