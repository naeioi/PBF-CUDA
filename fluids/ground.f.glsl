# version 330 core

in vec2 texCoord;
out vec4 FragColor;

void main() {
	float scale = 10;
	vec2 texCoord2 = texCoord * 10;
	float x = mod(texCoord2.x, 1), y = mod(texCoord2.y, 1);
	int flip = 1;
	if (x > 0.5) flip = 1 - flip;
	if (y > 0.5) flip = 1 - flip;

	if (flip == 1)
		FragColor = vec4(0.8, 0.8, 0.8, 1);
	else
		FragColor = vec4(0.6, 0.6, 0.6, 1);
}