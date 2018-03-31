#include <cstdio>
#include <cstdlib>

void fexit(const int code = -1, const char* msg = nullptr) {
	if (msg)
		fprintf(stderr, msg);
	return exit(code);
}