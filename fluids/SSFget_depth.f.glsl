# version 330 core

uniform float p_n;
uniform float p_f;

in vec4 viewPos;
in vec4 projPos;
out vec4 FragColor;

float linearize(float d) {
	float f = p_f, n = p_n;
	return 2 * f * n / (d * (f - n) - (f + n));
}

void main() {

	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) discard;

	// gl_FragCoord.z is NOT projPos.x / projPos.w!
	// FragColor.r = -linearize(projPos.z / projPos.w);
	FragColor.r = -viewPos.z;
}