#include <cstdio>
#include <cstdlib>
#include <glad/glad.h>

void fexit(const int code = -1, const char* msg = nullptr) {
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