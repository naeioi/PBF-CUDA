# version 420
#extension GL_EXT_shader_image_load_store : enable
uniform float k;
uniform float p_f;
uniform float p_n;
uniform int s_w;
uniform int s_h;
float l_n;

uniform sampler2D hTex;
uniform layout(r32f) image2D zImg;

float linearize(float d) {
	float f = p_f, n = p_n;
	return l_n / (d * (f - n) - (f + n));
}

float proj(float d) {
	float f = p_f, n = p_n;
	return (f + n) / (f - n) + 2 * f*n / ((f - n) * d);
}

float getZ(int x, int y) {
	return -linearize(imageLoad(zImg, ivec2(x, y)).x);
}

void setZ(int x, int y, float z) {
	imageStore(zImg, ivec2(x, y), vec4(proj(-z), 0, 0, 0));
}

float getH(int x, int y) {
	return texture(hTex, vec2((x+0.5) / s_w, (y+0.5) / s_h)).x;
}

void main() {
	/* global */
	l_n = 2 * p_f * p_n;
	int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
	float z  = getZ(x, y);
	float zz = z + k * getH(x, y);
	setZ(x, y, zz);
}