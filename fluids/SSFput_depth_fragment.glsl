# version 330 core

in vec2 texCoord;
uniform sampler2D depthTex;
out vec4 FragColor;

void main() {

	vec3 depth = texture(depthTex, texCoord);
	/* Make sure background not hidden. 
	 * But depth value of background will still be overriden by quad 
	 */
	if (depth >= 1.0) discard;
	FragColor = vec4(col, 1.0);

}