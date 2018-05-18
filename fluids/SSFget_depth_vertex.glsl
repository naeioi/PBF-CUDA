# version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 proj;
uniform mat4 view;
uniform float pointRadius;

out vec4 clipPos;

void main() {
	clipPos = view * vec4(aPos, 1.0);
	float dist = length(vec3(clipPos));
	gl_PointSize = pointRadius / dist;
	gl_Position = proj * clipPos;
}