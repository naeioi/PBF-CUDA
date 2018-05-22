# version 330 core

layout(location = 0) out float H;

in vec2 texCoord;

const float THRE = 0.3;

uniform float p_n;
uniform float p_f;
uniform float p_t;
uniform float p_r;
uniform float s_w;
uniform float s_h;

uniform sampler2D zTex;
uniform sampler2D normalDTex;

float l_n;
float f_x, f_y, c_x, c_y;

float getZ(float x, float y) {
	return texture(zTex, vec2(x, y)).x;
}

vec3 getN(float x, float y) {
	return texture(normalDTex, vec2(x, y)).xyz;
}

float getD(float x, float y) {
	return texture(normalDTex, vec2(x, y)).w;
}

void main() {
	/* global */
	l_n = 2 * p_f * p_n;
	f_x = 2 * p_r / (s_w * p_n);
	f_y = 2 * p_t / (s_h * p_n);
	c_x = 2 / (s_w * f_x);
	c_y = 2 / (s_h * f_y);

	/* (x, y) in [-1, 1] */
	float x = gl_FragCoord.x / s_w, y = gl_FragCoord.y / s_h;
	float dx = 1 / s_w, dy = 1 / s_h;

	float z = getZ(x, y);
	float dzdx  = getZ(x + dx, y) - z, dzdy = getZ(x, y+dy) - z;
	float dzdx2 = z - getZ(x - dx, y), dzdy2 = z - getZ(x, y - dy);
	dzdx  = abs(dzdx)  > THRE ? 0 : dzdx;
	dzdy  = abs(dzdy)  > THRE ? 0 : dzdy;
	dzdx2 = abs(dzdx2) > THRE ? 0 : dzdx2;
	dzdy2 = abs(dzdy2) > THRE ? 0 : dzdy2;
	float d2zdx2 = dzdx - dzdx2, d2zdy2 = dzdy - dzdy2;

	float D = getD(x, y);
	float dDdx = getD(x + dx, y) - D, dDdy = getD(x, y + dy) - D;
	
	float Ex = 0.5 * dzdx * dDdx - d2zdx2 * D;
	float Ey = 0.5 * dzdy * dDdy - d2zdy2 * D;

	H = (c_y * Ex + c_x * Ey) * pow(D, -1.5);
}