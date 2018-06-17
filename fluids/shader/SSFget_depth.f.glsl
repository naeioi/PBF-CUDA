# version 330 core

uniform float p_n;
uniform float p_f;
uniform mat4 proj;
uniform float r;

in vec4 viewPos;
in vec4 projPos;

/* TODO: add depth & thickness drawbuffer */
layout(location = 0) out vec4 FragColor;

float linearize(float d) {
	float f = p_f, n = p_n;
	return 2 * f * n / (d * (f - n) - (f + n));
}

void main() {

	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) {
		discard;
		return;
	}

	vec4 nviewPos = vec4(viewPos.xyz + vec3(x, y, z) * r, 1);
	vec4 nclipPos = proj * nviewPos;
	float nz_ndc = nclipPos.z / nclipPos.w;
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * nz_ndc + gl_DepthRange.far + gl_DepthRange.near);

	// gl_FragCoord.z is NOT projPos.x / projPos.w!
	// Write to d_depth_r
	FragColor.r = -nviewPos.z;
}