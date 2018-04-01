#include "FluidSystem.h"

int main() {
	FluidSystem fluids;

	fluids.initSource();

	while (1) {
		// fluids.stepSource();
		// fluids.stepSimulate();
		fluids.render();
	}
}