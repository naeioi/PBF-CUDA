#include "Camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/common.hpp>

Camera::Camera()
{
}

Camera::Camera(const glm::vec3 & pos, float aspect)
{
	*this = Camera(pos, -pos, glm::vec3(0.f, 0.f, 1.f), 60.f, aspect);
}

Camera::Camera(const glm::vec3 & pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect)
	: pos(pos)
	, moveFront(glm::normalize(front))
	, lookFront(glm::normalize(front))
	, up(glm::normalize(up))
	, ofov(fov), fov(fov)
	, aspect(aspect)
	, yaw(0.f), pitch(0.f)
	, isSyncMoveAndLook(false)
{
}


Camera::~Camera()
{
}

void Camera::use(const Shader & shader) const
{
	glm::mat4 view = glm::lookAt(pos, pos + lookFront, up);
	glm::mat4 pers = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
	shader.setUnif("view", view);
	shader.setUnif("proj", pers);
}

void Camera::move(const glm::vec3 & step)
{
	glm::vec3 right = glm::cross(moveFront, up);
	pos += step.x * right + step.y * up + step.z * moveFront;
}

void Camera::absMove(const glm::vec3 & step)
{
	pos += step;
}

void Camera::incrYaw(float degree)
{
	setYaw(yaw + degree);
}

void Camera::incrPitch(float degree)
{
	setPitch(pitch + degree);
}

void Camera::setYaw(float degree)
{
	yaw = degree;
	if (yaw > 89.f) yaw = 89.f;
	if (yaw < -89.f) yaw = -89.f;
	updateLookFront();
}

void Camera::setPitch(float degree)
{
	pitch = degree;
	if (pitch > 89.f) pitch = 89.f;
	if (pitch < -89.f) pitch = -89.f;
	updateLookFront();
}

void Camera::setAspect(float aspect_)
{
	aspect = aspect_; 
}

void Camera::zoomIn(float scale)
{
	float nfov = (float) glm::degrees(2 * atan(tan(glm::radians(fov) * 0.5) / scale));
	if (nfov < 1.f) nfov = 1.f;
	if (nfov > 45.f) nfov = 45.f;
	fov = nfov;
}

void Camera::zoom(float scale)
{
	float nfov = (float) glm::degrees(2 * atan(tan(glm::radians(ofov) * 0.5) / scale));
	if (nfov < 1.f) nfov = 1.f;
	if (nfov > 45.f) nfov = 45.f;
	fov = nfov;
}

bool Camera::toggleSyncMoveAndLook() {
	return isSyncMoveAndLook = !isSyncMoveAndLook;
}

void Camera::setSyncMoveAndLook(bool sync)
{
	isSyncMoveAndLook = sync;
}

void Camera::updateLookFront() {
	float rpitch = glm::radians(pitch), ryaw = glm::radians(yaw);
	glm::vec3 right = glm::cross(moveFront, up);
	glm::vec3 local = glm::vec3(
		cos(rpitch) * sin(ryaw),
		sin(rpitch),
		cos(rpitch) * cos(ryaw)
	);
	lookFront = local.x * right + local.y * up + local.z * moveFront;

	if (isSyncMoveAndLook) {
		moveFront = lookFront;
		yaw = pitch = 0;
	}
}

const glm::vec3& Camera::getPos() const {
	return pos;
}