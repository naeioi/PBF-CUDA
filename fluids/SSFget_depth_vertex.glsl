# version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 proj;
uniform mat4 view;
uniform float pointRadius;

void main() {
	vec4 eyePos = view * vec4(aPos, 1.0);
	float dist = length(vec3(eyePos));
	gl_PointSize = pointRadius / dist;
	gl_Position = proj * eyePos;
}