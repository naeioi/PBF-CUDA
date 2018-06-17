# version 330 core

uniform vec4 color;
in vec4 posColor;
out vec4 FragColor;

void main() {
	FragColor = posColor;
}