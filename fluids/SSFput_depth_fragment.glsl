# version 330 core

in vec2 texCoord;
uniform sampler2D depthTex;
out vec4 FragColor;

float linearizeDepth(float d) {
	/* TODO */
	return d;
}

void main() {

	float depth = texture(depthTex, texCoord).x;
	float ldepth = linearizeDepth(depth);
	/* Make sure background not hidden. 
	 * But depth value of background will still be overriden by quad 
	 */
	if (depth >= 1.0) discard;
	FragColor = vec4(ldepth, ldepth, ldepth , 1.0);

}