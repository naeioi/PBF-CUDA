#pragma once
#include "ParticleSource.h"
#include <cstdlib>
class DoubleDamSource :
	public ParticleSource
{
public:
	DoubleDamSource(float3 ulim1, float3 llim1, int3 ns1, float3 ulim2, float3 llim2, int3 ns2)
		: m_ulim1(ulim1), m_ulim2(ulim2), m_llim1(llim1), m_llim2(llim2), m_ns1(ns1), m_ns2(ns2), m_count(0)
		, m_pos(nullptr), m_vel(nullptr), m_iid(nullptr), m_nallocated(0) {
		m_d1 = ulim1 - llim1;
		m_d1.x /= ns1.x;
		m_d1.y /= ns1.y;
		m_d1.z /= ns1.z;

		m_d2 = ulim2 - llim2;
		m_d2.x /= ns2.x;
		m_d2.y /= ns2.y;
		m_d2.z /= ns2.z;
	};
	~DoubleDamSource() {
		if (m_pos) free(m_pos);
		if (m_vel) free(m_vel);
		if (m_iid) free(m_iid);
	};

	/* return number of particles */
	int initialize(uint, uint, uint, int);
	int update(uint, uint, uint, int);
	int reset(uint, uint, uint, int);

private:
	float3 m_ulim1, m_llim1, m_ulim2, m_llim2;
	int3 m_ns1, m_ns2;
	int m_count;

	float3 m_d1, m_d2;

	/* memory policy:
	* Generate data on memory fisrt,
	* Then (partially) transfer to GPU memory.
	*/
	float3 *m_pos, *m_vel;
	uint *m_iid;
	int m_nallocated;

	void __realloc(int);
	void generate_cube(float3 ulim, float3 llim, float3 d, int3 ns);
};

