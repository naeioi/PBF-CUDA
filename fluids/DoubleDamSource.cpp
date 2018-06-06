#include "DoubleDamSource.h"
#include <glad\glad.h>
#include <cstdlib>

void DoubleDamSource::generate_cube(float3 ulim, float3 llim, float3 d, int3 ns) {
	float sx = llim.x + d.x / 2, sy = llim.y + d.y / 2, sz = llim.z + d.z / 2, x, y, z;

	x = sx;
	for (int i = 0; i < ns.x; i++, x += d.x) {
		y = sy;
		for (int j = 0; j < ns.y; j++, y += d.y) {
			z = sz;
			for (int k = 0; k < ns.z; k++, z += d.z, m_count++) {
				float r1 = 1.f * rand() / RAND_MAX, r2 = 1.f * rand() / RAND_MAX, r3 = 1.f * rand() / RAND_MAX;
				m_pos[m_count] = make_float3(x, y, z) + 0.1f * make_float3(sx * r1, sy * r2, sz * r3);
				m_vel[m_count] = make_float3(0.f, 0.f, 0.f);
				m_iid[m_count] = m_count;
			}
		}
	}
}

int DoubleDamSource::initialize(uint pos, uint vel, uint iid, int max_nparticle) {

	srand(27);

	m_count = 0;
	__realloc(max_nparticle);

	generate_cube(m_ulim1, m_llim1, m_d1, m_ns1);
	generate_cube(m_ulim1, m_llim2, m_d2, m_ns2);

	glBindBuffer(GL_ARRAY_BUFFER, pos);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_pos[0]), m_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vel);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_vel[0]), m_vel);
	glBindBuffer(GL_ARRAY_BUFFER, iid);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_count * sizeof(m_iid[0]), m_iid);

	return m_count;
}

int DoubleDamSource::update(uint pos, uint vel, uint iid, int max_nparticle) {
	return m_count;
}

int DoubleDamSource::reset(uint pos, uint vel, uint iid, int max_nparticle) {
	return initialize(pos, vel, iid, max_nparticle);
}

void DoubleDamSource::__realloc(int max_nparticle)
{
	if (m_nallocated < max_nparticle) {
		if (m_pos) {
			m_pos = (float3*)realloc(m_pos, max_nparticle * sizeof(m_pos[0]));
			m_vel = (float3*)realloc(m_vel, max_nparticle * sizeof(m_vel[0]));
			m_iid = (uint*)realloc(m_iid, max_nparticle * sizeof(m_iid[0]));
		}
		else {
			m_pos = (float3*)malloc(max_nparticle * sizeof(m_pos[0]));
			m_vel = (float3*)malloc(max_nparticle * sizeof(m_vel[0]));
			m_iid = (uint*)malloc(max_nparticle * sizeof(m_iid[0]));
		}

		m_nallocated = max_nparticle;
	}
}
