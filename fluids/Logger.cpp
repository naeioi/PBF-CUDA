#include "Logger.h"
#include <GLFW\glfw3.h>
#include <cstdio>
#include <algorithm>
using namespace std;

Logger* Logger::m_instance = nullptr;

Logger& Logger::getInstance() {
	if (!m_instance) {
		m_instance = new Logger();
	}

	return *m_instance;
}

#define LOG(m, v) \
	if (type == m##_START) v##_bg = now; \
	if (type == m##_END) { \
		double d = now - v##_bg; \
		v##_sum += d; \
		v##_max = max(v##_max, d); \
		v##_min = min(v##_min, d); \
	}
	

void Logger::logTime(TType type) {
	if (!enableLogTime) return;

	if (type == FRAME_START) nframe++;

	double now = glfwGetTime();
	LOG(SIMULATE, simu)
	LOG(RENDER, render)
	LOG(ADVECT, advect)
	LOG(GRID, grid)
	LOG(DENSITY, density)
	LOG(VELOCITY_UPDATE, vel_upd)
	LOG(VELOCITY_CORRECT, vel_corr)
}

#define RESET(v) v##_sum = 0., v##_max = -1., v##_min = 1e38
Logger::Logger()
{
	enableLogTime = false;
	nframe = 0;
	RESET(simu);
	RESET(render);
	RESET(advect);
	RESET(grid);
	RESET(density);
	RESET(vel_upd);
	RESET(vel_corr);
}

#define REPORT(v) printf(#v"\t%.1f/%.1f/%.1f/%.1f\n", v##_sum, 1000*v##_sum/nframe, 1000*v##_min, 1000*v##_max)
void Logger::report() {
	printf("-- Timing report --\n");
	REPORT(simu);
	REPORT(render);
	REPORT(advect);
	REPORT(grid);
	REPORT(density);
	REPORT(vel_upd);
	REPORT(vel_corr);
}

Logger::~Logger()
{
}
