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

float f_x, f_y, c_x, c_y;

float getZ(float x, float y) {
	return texture(zTex, vec2(x, y)).x;
}

vec3 getN(int x, int y) {
	return texelFetch(normalDTex, ivec2(x, y), 0).xyz;
}

float getD(float x, float y) {
	return texture(normalDTex, vec2(x, y)).w;
}

void method1() {
	/* (x, y) in [0, 1] */
	int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
	float dx = 1 / s_w, dy = 1 / s_h;

	vec3 n = getN(x, y);
	float nx = n.x, ny = n.y;
	float dnxdx = getN(x + 1, y).x - nx, dnydy = getN(x, y + 1).y - ny;
	float dnxdx2 = nx - getN(x - 1, y).x, dnydy2 = ny - getN(x, y - 1).y;

	if (abs(dnxdx2) < abs(dnxdx)) dnxdx = dnxdx2;
	if (abs(dnydy2) < abs(dnydy)) dnydy = dnydy2;

	H = dnxdx + dnydy;
}

void method2() {
	/* (x, y) in [0, 1] */
	int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
	float dx = 1 / s_w, dy = 1 / s_h;

	float z = getZ(x, y);
	float dzdx = getZ(x + dx, y) - z, dzdy = getZ(x, y + dy) - z;
	float dzdx2 = z - getZ(x - dx, y), dzdy2 = z - getZ(x, y - dy);
	dzdx = abs(dzdx)  > THRE ? 0 : dzdx;
	dzdy = abs(dzdy)  > THRE ? 0 : dzdy;
	dzdx2 = abs(dzdx2) > THRE ? 0 : dzdx2;
	dzdy2 = abs(dzdy2) > THRE ? 0 : dzdy2;
	float d2zdx2 = dzdx - dzdx2, d2zdy2 = dzdy - dzdy2;

	float D = getD(x, y);
	float dDdx = getD(x + dx, y) - D, dDdy = getD(x, y + dy) - D;

	float Ex = 0.5 * dzdx * dDdx - d2zdx2 * D;
	float Ey = 0.5 * dzdy * dDdy - d2zdy2 * D;

	H = (c_y * Ex + c_x * Ey) * pow(D, -1.5);
}

void main() {
	/* global */
	f_x = p_n / p_r;
	f_y = p_n / p_t;
	c_x = 2 / (s_w * f_x);
	c_y = 2 / (s_h * f_y);
		
	method1();
}