#pragma once
#include "ParticleSource.h"
#include <cstdlib>

class FixedCubeSource :
	public ParticleSource
{
public:
	FixedCubeSource(float3 ulim, float3 llim, int3 ns) 
		: m_ulim(ulim), m_llim(llim), m_ns(ns), m_count(0)
		, m_pos(nullptr), m_vel(nullptr), m_iid(nullptr), m_nallocated(0) {
		m_d = ulim - llim;
		m_d.x /= ns.x;
		m_d.y /= ns.y;
		m_d.z /= ns.z;
	};
	~FixedCubeSource() {
		if (m_pos) {
			free(m_pos);
			free(m_vel);
			free(m_iid);
		}
	};

	/* return number of particles */
	int initialize(uint, uint, uint, int);
	int update(uint, uint, uint, int);
	int reset(uint, uint, uint, int);

private:
	float3 m_ulim, m_llim;
	int3 m_ns;
	int m_count;

	float3 m_d;

	/* memory policy: 
	 * Generate data on memory fisrt, 
	 * Then (partially) transfer to GPU memory.
	 */
	float3 *m_pos, *m_vel;
	uint *m_iid;
	int m_nallocated;

	void __realloc(int);
};

