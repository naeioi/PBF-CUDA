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

uniform sampler2D zTex;

float l_n;
float f_x, f_y, c_x, c_y, c_x2, c_y2;

float getZ(float x, float y) {
	return texture(zTex, vec2(x, y)).x;
}

void main() {
	/* global */
	l_n = 2 * p_f * p_n;
	f_x = 2 * p_r / (s_w * p_n);
	f_y = 2 * p_t / (s_h * p_n);
	c_x = 2 / (s_w * f_x);
	c_y = 2 / (s_h * f_y);
	c_x2 = c_x * c_x;
	c_y2 = c_y * c_y;

	/* (x, y) in [-1, 1] */
	float x = gl_FragCoord.x / s_w, y = gl_FragCoord.y / s_h;
	float dx = 1 / s_w, dy = 1 / s_h;
	float z = getZ(x, y), z2 = z * z;
	float dzdx = getZ(x + dx, y) - z, dzdy = getZ(x, y+dy) - z;
	
	if (abs(dzdx) > THRE) dzdx = 0;
	if (abs(dzdy) > THRE) dzdy = 0;
	
	float dzdx2 = dzdx * dzdx, dzdy2 = dzdy * dzdy;
	vec3 n = vec3(-c_y * dzdx, -c_x * dzdy, c_x*c_y*z);
	float d = length(n);
	normal_D = vec4(n / d, d);
}