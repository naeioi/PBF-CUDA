#include "FluidSystem.h"
#include "Logger.h"
#include <conio.h>

extern bool move;

int main() {
	FluidSystem fluids;

	fluids.initSource();

	while (1) {
		Logger::getInstance().logTime(Logger::FRAME_START);
		Logger::getInstance().logTime(Logger::RENDER_START);
		fluids.render();
		Logger::getInstance().logTime(Logger::RENDER_END);
		if (1) {
			Logger::getInstance().logTime(Logger::SIMULATE_START);
			fluids.stepSimulate();
			Logger::getInstance().logTime(Logger::SIMULATE_END);
			move = false;
		}
	}
}