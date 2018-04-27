#define STRINGIFY(s) #s

/* Stringify does not support # and newline, so hack outside out it */

/* Shader for box */
const char* box_vshader = "# version 330 core\n" STRINGIFY(

layout (location = 0) in vec3 aPos;

uniform mat4 proj; 
uniform mat4 view; 

//	out vec4 FragPos;	

void main() {
	vec4 FragPos = view * vec4(aPos, 1.0);
	gl_Position = proj * FragPos;
}

);

const char* box_fshader = "# version 330 core\n" STRINGIFY(

uniform vec4 color;
out vec4 FragColor;

void main() {
	FragColor = color;
}

);

/* Shader for particle */

const char* particle_vshader = "# version 330 core\n" STRINGIFY(

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

);

const char* particle_fshader = "# version 330 core\n" STRINGIFY(

uniform vec4 color;
uniform uint hlIndex;
flat in uint iid;
out vec4 FragColor;

void main() {
	vec3 lightDir = normalize(vec3(1, -1, 1));
	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) discard;
	float r = iid % 16u / 16.f;
	float g = iid / 16u % 16u / 16.f;
	float b = iid / 16u % 16u / 16.f;
	vec4 rgba = vec4(dot(lightDir, vec3(x, y, z)) * vec3(r, b, 0), 1);
	vec4 white = vec4(dot(lightDir, vec3(x, y, z)) * vec3(1, 1, 1), 1) + 0.2;
	if (iid == hlIndex)
		FragColor = white;
	else
		FragColor = rgba;
	// FragColor = vec4(gl_FragCoord.z, 0, 0, 1);
	// FragColor = vec4(dot(lightDir, vec3(x, y, z)) * vec3(r, b, 0), 1);
	// FragColor = dot(lightDir, vec3(x, y, z)) * vec3(r, b, 0, 1) * 0.7;
}

);