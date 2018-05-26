#include "Camera.h"
#include "Input.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/common.hpp>
#include <glm/gtx/rotate_vector.hpp>

Camera::Camera()
{
}

Camera::Camera(const glm::vec3 & pos, const glm::vec3 &focus, float aspect)
{
	glm::vec3 front(focus - pos), up = glm::vec3(0.f, 0.f, 1.f);
	up = glm::cross(front, glm::cross(up, front));
	*this = Camera(pos, front, up, 60.f, aspect);
}

Camera::Camera(const glm::vec3 & pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect)
	: pos(pos)
	, front(front)
	, up(glm::normalize(up))
	, fov(fov)
	, aspect(aspect)
{
	rotx = glm::vec3(0.f, 0.f, 1.f);
	roty = glm::normalize(glm::cross(front, rotx));
}


Camera::~Camera()
{
}

void Camera::use(const Shader & shader, bool translate_invariant) const
{
	glm::mat4 view = glm::lookAt(pos, pos + front, up);
	if (translate_invariant) {
		view = glm::mat4(glm::mat3(view));
	}
	glm::mat4 pers = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
	shader.setUnif("view", view);
	shader.setUnif("proj", pers);
}

void Camera::setUp(const glm::vec3 &up_) { up = up_; }
void Camera::setPos(const glm::vec3 &pos_) { pos = pos_; }
void Camera::setFront(const glm::vec3 &front_) { front = front_; }

void Camera::setAspect(float aspect_)
{
	aspect = aspect_; 
}

void Camera::rotate(const glm::vec2 dxy) {

	glm::vec3 center = pos + front;
	/* For horizontal panning, rotate camera within plane perpendicular to `up' direction */
	if (dxy.x != 0) {
		const glm::vec3 &axis = rotx;
		/* for now, manually update pos, front and up in renderer */
		front = glm::rotate(front, -dxy.x * Input::SCREEN_ROTATE_RATE, axis);
		up = glm::rotate(up, -dxy.x * Input::SCREEN_ROTATE_RATE, axis);
		pos = center - front;

		roty = glm::rotate(roty, -dxy.x * Input::SCREEN_ROTATE_RATE, axis);
	}
	/* For verticle panning, rotate camera within plane perpendicular to cross(up, front) direction */
	if (dxy.y != 0) {
		const glm::vec3 &axis = roty;

		front = glm::rotate(front, -dxy.y * Input::SCREEN_ROTATE_RATE, axis);
		up = glm::rotate(up, -dxy.y * Input::SCREEN_ROTATE_RATE, axis);
		pos = center - front;
	}

}

void Camera::pan(const glm::vec2 dxy) {
	glm::vec3 cam_d = dxy.x * -glm::normalize(glm::cross(front, up)) + dxy.y * glm::normalize(up);
	pos += Input::SCREEN_PAN_RATE * cam_d * glm::length(front);
}

void Camera::zoom(float dy)
{
	const float min_d = 0.1f, max_d = 10.f;
	if (dy > 0) {
		if (front.length() < min_d) return;
		pos += front * Input::SCREEN_SCROLL_RATE;
		front -= front * Input::SCREEN_SCROLL_RATE;
	}
	else {
		if (front.length() > max_d) return;
		pos -= front * Input::SCREEN_SCROLL_RATE;
		front += front * Input::SCREEN_SCROLL_RATE;
	}
}

ProjectionInfo Camera::getProjectionInfo() const
{
	ProjectionInfo i;
	float tanHalfFov = tan(glm::radians(fov) * 0.5f);
	i.n = 0.1f;
	i.f = 100.f;
	i.t = tanHalfFov * i.n;
	i.b = -i.t;
	i.r = aspect * i.t;
	i.l = -i.r;
	return i;
}
