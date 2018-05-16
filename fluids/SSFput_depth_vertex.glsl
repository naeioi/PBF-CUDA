# version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 proj;
uniform mat4 view;
uniform float pointRadius;

out vec2 texCoord;

void main() {
	vec4 eyePos = view * vec4(aPos, 1.0);
	float dist = length(vec3(eyePos));
	gl_PointSize = pointRadius / dist;
	gl_Position = proj * eyePos;
	texCoord = aTexCoord;
}