# version 330 core

uniform vec4 color;
uniform uint hlIndex;
flat in uint iid;
out vec4 FragColor;

void hsl_shading() {
	const uint hsl_loop = 360u;
	// float hsl_h = mod(iid, hsl_loop);
	// const float hsl_s = 1;
	float hsl_h = 220;
	float hsl_s = 1;
	float hsl_l = 0.5 + 0.4 * mod(iid, hsl_loop) / hsl_loop;

	float hsl_hp = hsl_h / 60.0f;
	float hsl_c = hsl_s * (1 - abs(2 * hsl_l - 1));
	float hsl_x = hsl_c * (1 - abs(mod(hsl_hp, 2) - 1));

	vec3 rgb = vec3(1, 1, 1);
	if (0 <= hsl_hp && hsl_hp <= 1)		 rgb = vec3(hsl_c, hsl_x, 0);
	else if (1 <= hsl_hp && hsl_hp <= 2) rgb = vec3(hsl_x, hsl_c, 0);
	else if (2 <= hsl_hp && hsl_hp <= 3) rgb = vec3(0, hsl_c, hsl_x);
	else if (3 <= hsl_hp && hsl_hp <= 4) rgb = vec3(0, hsl_x, hsl_c);
	else if (4 <= hsl_hp && hsl_hp <= 5) rgb = vec3(hsl_x, 0, hsl_c);
	else if (5 <= hsl_hp && hsl_hp <= 6) rgb = vec3(hsl_c, 0, hsl_x);
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
}

void random_shading() {
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
}

void main() {

	hsl_shading();
	
}