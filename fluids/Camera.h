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
	void setAspect(float aspect);
	void setPos(const glm::vec3 &pos);
	void setFront(const glm::vec3 &front);
	void setUp(const glm::vec3 &up);

	void rotate(const glm::vec2 dxy);
	void pan(const glm::vec2 dxy);
	void zoom(float dy);

	const glm::vec3& getPos() const { return pos; }
	const glm::vec3& getUp() const { return up;  }
	/* Convection: len(front) == len(lookat center - pos) 
	 * or simply let front = lookat - pos
	 * This convection makes camera rotation work properly.
	 */
	const glm::vec3& getFront() const { return front;  }

private:
	glm::vec3 pos;
	glm::vec3 up;
	glm::vec3 front;

	/* Rotate axis when pan horizontally and vertically on screen */
	glm::vec3 rotx, roty;

	float fov;
	float ofov;
	float aspect;
};

