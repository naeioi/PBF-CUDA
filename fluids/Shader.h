#pragma once

#include "helper.h"
#include <glad\glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <cstring>
#include <cstdio>

extern const char* vshader;
extern const char* fshader;
 
class Shader
{
public:
	uint id;
	Shader();
	Shader(const char *vshader, const char *fshader);
	~Shader();

	bool loaded();
	void use();
	static Shader& now();

	void setUnif(const std::string &name, bool value) const;
	void setUnif(const std::string &name, int value) const;
	void setUnif(const std::string &name, float value) const;
	void setUnif(const std::string &name, double value) const;
	void setUnif(const std::string &name, glm::mat2 &mat) const;
	void setUnif(const std::string &name, glm::mat3 &mat) const;
	void setUnif(const std::string &name, glm::mat4 &mat) const;
	void setUnif(const std::string &name, glm::vec2 &vec) const;
	void setUnif(const std::string &name, glm::vec3 &vec) const;
	void setUnif(const std::string &name, glm::vec4 &vec) const;

private:
	static Shader* current;

};
