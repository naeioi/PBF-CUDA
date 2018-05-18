#include <cstdio>
#include <cstdlib>
#include <glad/glad.h>
#include "helper.h"

void fexit(const int code, const char* msg) {
	if (msg)
		fprintf(stderr, msg);
	return exit(code);
}

void checkGLErr() {
	GLenum err;
	const char *errString;
	if ((err = glGetError()) != GL_NO_ERROR) {
		switch (err) {
			case GL_INVALID_OPERATION:      errString = "INVALID_OPERATION";      break;
			case GL_INVALID_ENUM:           errString = "INVALID_ENUM";           break;
			case GL_INVALID_VALUE:          errString = "INVALID_VALUE";          break;
			case GL_OUT_OF_MEMORY:          errString = "OUT_OF_MEMORY";          break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:  errString = "INVALID_FRAMEBUFFER_OPERATION";  break;
		}
		fprintf(stderr, "OpenGL Error #%d: %s\n", err, errString);
		fexit(-1);
	}
}

void checkFramebufferComplete()
{
	GLenum err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	const char *errString = NULL;
	switch (err) {
	case GL_FRAMEBUFFER_UNDEFINED: errString = "GL_FRAMEBUFFER_UNDEFINED"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: errString = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: errString = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: errString = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: errString = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER"; break;
	case GL_FRAMEBUFFER_UNSUPPORTED: errString = "GL_FRAMEBUFFER_UNSUPPORTED"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: errString = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: errString = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS"; break;
	}

	if (errString) {
		fprintf(stderr, "OpenGL Framebuffer Error #%d: %s\n", err, errString);
		fexit(-1);
	}
	else {
		printf("Framebuffer complete check ok\n");
	}
}
