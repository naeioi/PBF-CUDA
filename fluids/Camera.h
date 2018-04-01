#pragma once
#include "Shader.h"
#include <glm/glm.hpp>

class Camera
{
public:
	Camera();
	/* Specify only camera position and looks at origin */
	Camera(const glm::vec3 &pos, float aspect);
	Camera(const glm::vec3 &pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect);
	~Camera();

	void use(const Shader &shader) const;
	void move(const glm::vec3 &step);
	void absMove(const glm::vec3 &step);
	void incrYaw(float degree);
	void incrPitch(float degree);
	void setYaw(float degree);
	void setPitch(float degree);
	void setAspect(float aspect);
	void zoomIn(float scale);
	void zoom(float scale);

	const glm::vec3& getPos() const;

	void setSyncMoveAndLook(bool sync);
	bool toggleSyncMoveAndLook();

private:
	glm::vec3 pos;
	glm::vec3 moveFront; 
	glm::vec3 up;
	glm::vec3 lookFront;

	void updateLookFront();

	float yaw;
	float pitch;
	float roll;

	float fov;
	float ofov;
	float aspect;

	bool isSyncMoveAndLook;
};

