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

float getZ(int x, int y) {
	return imageLoad(zImg, ivec2(x, y)).x;
}

void setZ(int x, int y, float z) {
	imageStore(zImg, ivec2(x, y), vec4(z, 0, 0, 0));
}

float getH(int x, int y) {
	return texelFetch(hTex, ivec2(x, y), 0).x;
}

void main() {
	int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
	float z  = getZ(x, y);
	if (z > 99) return;
	float zz = z - k * getH(x, y);
	setZ(x, y, zz);
}