# version 420
#extension GL_EXT_shader_image_load_store : enable

uniform float p_f;
uniform float p_n;
uniform int s_w;
uniform int s_h;
uniform int kernel_r;
uniform float blur_r;
uniform float blur_z;

/* zA: source depth map, zB: target depth map */
uniform sampler2D zA;
uniform layout(r32f) image2D zB;

void setZ(int x, int y, float z) {
	imageStore(zB, ivec2(x, y), vec4(z, 0, 0, 0));
}

float getZ(int x, int y) {
	return texelFetch(zA, ivec2(x, y), 0).x;
}

float bilateral(int x, int y) {
	float z = getZ(x, y);
	float sum = 0, wsum = 0;

	for(int dx = -kernel_r; dx <= kernel_r; dx++)
		for (int dy = -kernel_r; dy <= kernel_r; dy++) {
			float s = getZ(x+dx, y+dy);

			float w = exp(- (dx*dx + dy*dy) * blur_r * blur_r);

			float r2 = (s - z) * blur_z;
			float g = exp(-r2 * r2);

			float wg = w * g;
			sum += s * wg;
			wsum += wg;
		}

	if (wsum > 0) sum /= wsum;
	return sum;
}

float gaussian(int x, int y) {
	float z = getZ(x, y);
	float sum = 0, wsum = 0;

	for (int dx = -kernel_r; dx <= kernel_r; dx++)
		for (int dy = -kernel_r; dy <= kernel_r; dy++) {
			float s = getZ(x + dx, y + dy);
			float w = exp(-(dx*dx + dy * dy) * blur_r * blur_r);

			sum += s * w;
			wsum += w;
		}

	if (wsum > 0) sum /= wsum;
	return sum;
}

void main() {
	int x = int(gl_FragCoord.x), y = int(gl_FragCoord.y);
	float z = getZ(x, y);
	if (z > 99) return;

	float zz = gaussian(x, y);
	setZ(x, y, zz);
}