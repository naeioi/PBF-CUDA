# version 330 core

in vec4 clipPos;
out vec4 FragColor;
out float gl_FragDepth;

void main() {

	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) discard;

	// gl_FragDepth = -clipPos.z;
	// FragColor = vec4(gl_FragCoord.z, 0, 0, 1);
}