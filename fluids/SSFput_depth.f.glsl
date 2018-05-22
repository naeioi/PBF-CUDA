# version 330 core

in vec2 texCoord;
uniform float p_n;
uniform float p_f;
uniform float p_t;
uniform float p_r;

uniform sampler2D zTex;
out vec4 FragColor;

float proj(float ze) {
	return (p_f + p_n) / (p_f - p_n) + 2 * p_f*p_n / ((p_f - p_n) * ze);
}

void main() {

	// ze to z_ndc to gl_FragDepth
	// REF: https://computergraphics.stackexchange.com/questions/6308/why-does-this-gl-fragdepth-calculation-work?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	float ze = texture(zTex, texCoord).x;
	float z_ndc = proj(-ze);
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * z_ndc + gl_DepthRange.far + gl_DepthRange.near);
	float log_ze = log(ze);
	// FragColor = vec4(log_ze, log_ze, log_ze, 1.0);
	FragColor = vec4(ze, ze, ze, 1.0);
}