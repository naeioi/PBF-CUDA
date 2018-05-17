#include "Shader.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glad\glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <exception>
#include <fstream>
using namespace std;

Shader* Shader::current = nullptr;

Shader::Shader() {
}

Shader::Shader(const char* vshader, const char* fshader)
{
	
	const char* vShaderCode = vshader;
	const char* fShaderCode = fshader;

	// compile shaders
	unsigned int vertex, fragment;
	int success;
	char infoLog[512];

	// vertex Shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, nullptr);
	glCompileShader(vertex);
	// print compile errors if any
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, nullptr);
	glCompileShader(fragment);
	// print compile errors if any
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	id = glCreateProgram();
	glAttachShader(id, vertex);
	glAttachShader(id, fragment);
	glLinkProgram(id);
	// print linking errors if any
	glGetProgramiv(id, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(id, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

Shader::Shader(const Filename & vfile, const Filename & ffile)
{
	string vcode, fcode;

	{
		ifstream fin(vfile.path);
		if (!fin) fexit(-1, (string("File to open ") + vfile.path).c_str());
		stringstream ss;
		ss << fin.rdbuf();
		vcode = ss.str();
	}

	{
		ifstream fin(ffile.path);
		if (!fin) fexit(-1, (string("File to open ") + ffile.path).c_str());
		stringstream ss;
		ss << fin.rdbuf();
		fcode = ss.str();
	}

	*this = Shader(vcode.c_str(), fcode.c_str());
}


Shader::~Shader()
{
	
}

bool Shader::loaded() {
	return id != 0;
}

void Shader::use()
{
	glUseProgram(id);
	current = this;
}

void Shader::setUnif(const std::string & name, bool value) const
{
	setUnif(name, (int)value);
}

void Shader::setUnif(const std::string & name, int value) const
{
	glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}

void Shader::setUnif(const std::string & name, uint value) const
{
	glUniform1ui(glGetUniformLocation(id, name.c_str()), (uint)value);
}

void Shader::setUnif(const std::string & name, float value) const
{
	glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::setUnif(const std::string & name, double value) const
{
	glUniform1d(glGetUniformLocation(id, name.c_str()), value);
}

void Shader::setUnif(const std::string & name, glm::mat2 & mat) const
{
	/* TODO: glm::mat2 doesn't specify float or double but open requires */
	glUniformMatrix2fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::setUnif(const std::string & name, glm::mat3 & mat) const
{
	glUniformMatrix3fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::setUnif(const std::string & name, glm::mat4 & mat) const
{
	glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::setUnif(const std::string & name, glm::vec2 & vec) const
{
	glUniform2fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(vec));
}

void Shader::setUnif(const std::string & name, glm::vec3 & vec) const
{
	glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(vec));
}

void Shader::setUnif(const std::string & name, glm::vec4 & vec) const
{
	glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(vec));
}

Shader& Shader::now() {
	if (current == nullptr)
		/* use STL exception for now */
		throw std::exception("Not using any shader now!");
	return *current;
}