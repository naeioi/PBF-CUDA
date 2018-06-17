# version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 proj; 
uniform mat4 view; 

out vec3 posColor;

void main() {
	posColor = vec3(1, 1, 1) - clamp(aPos, vec3(0, 0, 0), vec3(1, 1, 1));
	vec4 FragPos = view * vec4(aPos, 1.0);
	gl_Position = proj * FragPos;
}
