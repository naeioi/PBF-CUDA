# version 330 core

layout(location = 0) out vec4 normal_D;

in vec2 texCoord;

const float THRE = 0.3;

uniform float p_n;
uniform float p_f;
uniform float p_t;
uniform float p_r;
uniform float s_w;
uniform float s_h;

uniform int keep_edge;

/* Assume left-hand coord: z > 0 for front of eye */
uniform sampler2D zTex;

float f_x, f_y, c_x, c_y, c_x2, c_y2;

/* return z in right-hand coord 
 * Because both opengl and paper assume right-hand coord
 * store as left-hand only for easy debug
 */
float getZ(float x, float y) {
	return -texture(zTex, vec2(x, y)).x;
}

void main() {
	/* global */
	f_x = p_n / p_r;
	f_y = p_n / p_t;
	c_x = 2 / (s_w * f_x);
	c_y = 2 / (s_h * f_y);
	c_x2 = c_x * c_x;
	c_y2 = c_y * c_y;

	/* (x, y) in [0, 1] */
	float x = texCoord.x, y = texCoord.y;
	float dx = 1 / s_w, dy = 1 / s_h;
	float z = getZ(x, y), z2 = z * z;
	float dzdx = getZ(x + dx, y) - z, dzdy = getZ(x, y+dy) - z;
	float dzdx2 = z - getZ(x - dx, y), dzdy2 = z - getZ(x, y - dy);
	
	/* Skip silhouette */
	if (keep_edge == 1) {
		if (abs(dzdx2) < abs(dzdx)) dzdx = dzdx2;
		if (abs(dzdy2) < abs(dzdy)) dzdy = dzdy2;
	}

	vec3 n = vec3(-c_y * dzdx, -c_x * dzdy, c_x*c_y*z);
	/* revert n.z to positive for debugging */
	n.z = -n.z;

	float d = length(n);
	normal_D = vec4(n / d, d);
}