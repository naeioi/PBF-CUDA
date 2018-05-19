# version 330 core

in vec2 texCoord;

uniform float p_n;
uniform float p_f;

uniform sampler2D zTex;

float linearize(float d) {
	float f = p_f, n = p_n;
	return 2 * f * n / (d * (f - n) - (f + n));
}

void main() {

	float depth = texture(depthTex, texCoord).x;
	float ldepth = -linearize(depth);
	/* Make sure background not hidden.
	* But depth value of background will still be overriden by quad
	*/
	if (depth >= 1.0) discard;
	FragColor = vec4(ldepth, ldepth, ldepth, 1.0);

}