#include "GUIParams.h"

GUIParams* GUIParams::instance = 0;

GUIParams & GUIParams::getInstance()
{
	if (instance == 0) {
		instance = new GUIParams();
	}

	return *instance;
}
