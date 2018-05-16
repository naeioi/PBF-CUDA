# version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in uint aiid;

uniform mat4 proj;
uniform mat4 view;
uniform float pointRadius;
uniform float pointScale;
uniform uint hlIndex;

//	out vec4 FragPos;	
flat out uint iid;

void main() {
	vec4 eyePos = view * vec4(aPos, 1.0);
	float dist = length(vec3(eyePos / eyePos.w));
	gl_PointSize = (aiid == hlIndex ? 2 : 1) * pointRadius * (pointScale / dist);
	gl_Position = proj * eyePos;
	iid = aiid;
}