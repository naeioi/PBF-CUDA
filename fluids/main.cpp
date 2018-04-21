#include "FluidSystem.h"
#include <conio.h>

extern bool move;

int main() {
	FluidSystem fluids;

	fluids.initSource();

	while (1) {
		fluids.render();
		// getch();
		if (1) {
			fluids.stepSimulate();
			move = false;
		}
	}
}