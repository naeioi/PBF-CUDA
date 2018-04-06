#include "FluidSystem.h"

extern bool move;

int main() {
	FluidSystem fluids;

	fluids.initSource();

	while (1) {
		// fluids.stepSource();
		if (1) {
			fluids.stepSimulate();
			move = false;
		}
		fluids.render();
	}
}