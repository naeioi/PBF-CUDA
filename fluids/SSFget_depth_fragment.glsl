# version 330 core

in vec4 clipPos;
out vec4 FragColor;
out float gl_FragDepth;

void main() {
	/* Nothing because only depth is needed */
	// gl_FragDepth = 0;
	FragColor = vec4(1, 0, 0, 1);
}