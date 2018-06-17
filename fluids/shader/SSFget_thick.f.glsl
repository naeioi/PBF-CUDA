# version 330 core

uniform float p_n;
uniform float p_f;
uniform mat4 proj;
uniform float r;

in vec4 viewPos;
in vec4 projPos;

/* TODO: add depth & thickness drawbuffer */
layout(location = 0) out float thickness;

void main() {

	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) {
		discard;
		return;
	}

	vec3 lightDir = vec3(0, 0, 1);
	thickness = 2 * r*dot(vec3(x, y, z), lightDir);
}