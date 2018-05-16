#define STRINGIFY(s) #s

/* Stringify does not support # and newline, so hack outside out it */

/* Shader for box */
const char* box_vshader = "# version 330 core\n" STRINGIFY(

layout (location = 0) in vec3 aPos;

uniform mat4 proj; 
uniform mat4 view; 

out vec3 posColor;
//	out vec4 FragPos;	

void main() {
	posColor = vec3(1, 1, 1) - clamp(aPos, vec3(0, 0, 0), vec3(1, 1, 1));
	vec4 FragPos = view * vec4(aPos, 1.0);
	gl_Position = proj * FragPos;
}

);

const char* box_fshader = "# version 330 core\n" STRINGIFY(

uniform vec4 color;
in vec4 posColor;
out vec4 FragColor;

void main() {
	FragColor = posColor;
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

	const float hsl_s = 0.5;
	const float hsl_l = 0.5;
	const uint hsl_loop = 360;

	uint hsl_h = mod(iid, hsl_loop);
	float hsl_hp = hsl_h / 60.0f;
	float hsl_c = hsl_s * (1 - abs(2 * hsl_l - 1));
	float hsl_x = hsl_c * (1 - abs(mod(hsl_hp, 2) - 1));
	
	vec3 rgb;
	if (0 <= hsl_hp && hsl_hp <= 1)		 rgb = vec3(hsl_c, hsl_x, 0);
	else if (1 <= hsl_hp && hsl_hp <= 2) rgb = vec3(hsl_x, hsl_c, 0);
	else if (2 <= hsl_hp && hsl_hp <= 3) rgb = vec3(0, hsl_c, hsl_x);
	else if (3 <= hsl_hp && hsl_hp <= 4) rgb = vec3(0, hsl_x, hsl_c);
	else if (4 <= hsl_hp && hsl_hp <= 5) rgb = vec3(hsl_x, 0, hsl_c);
	else if (5 <= hsl_hp && hsl_hp <= 5) rgb = vec3(hsl_c, 0, hsl_x);
	rgb += hsl_l - 0.5 * hsl_c;

	vec3 lightDir = normalize(vec3(1, -1, 1));
	float x = 2 * gl_PointCoord.x - 1;
	float y = 2 * gl_PointCoord.y - 1;
	float pho = x * x + y * y;
	float z = sqrt(1 - pho);
	if (pho > 1) discard;

	vec4 rgba = vec4(dot(lightDir, vec3(x, y, z)) * rgb, 1);
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