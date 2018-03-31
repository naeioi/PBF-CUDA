#pragma once
#include "helper.h"

class ParticleSource
{
public:
	ParticleSource() {};
	~ParticleSource() {};

	/* Params: vbo of pos, vel, iid + max_nparticle */
	virtual int initialize(uint, uint, uint, int) = 0;
	virtual int update(uint, uint, uint, int) = 0;
	virtual int reset(uint, uint, uint, int) = 0;

private:
	uint d_pos, d_vel, d_iid;
};

