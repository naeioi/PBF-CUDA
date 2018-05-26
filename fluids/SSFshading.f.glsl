# version 330 core

in vec2 texCoord;

uniform mat4 iview;

uniform float p_n;
uniform float p_f;
uniform float p_t;
uniform float p_r;
/* Schlick's approximation on Fresnel factor (reflection coef)
 * r0: Reflection coef when incoming light parallel to the normal 
 * r0 = [(n1 - n2)/(n1 + n2)]^2
 */
uniform float r0;

uniform sampler2D zTex;
uniform sampler2D normalDTex;
uniform samplerCube skyTex;

out vec4 FragColor;

float proj(float ze) {
	return (p_f + p_n) / (p_f - p_n) + 2 * p_f*p_n / ((p_f - p_n) * ze);
}

vec3 getPos() {
	/* Return in right-hand coord */
	float z = texture(zTex, texCoord).x;
	float x = texCoord.x, y = texCoord.y;
	x = (2 * x - 1)*p_r*z / p_n;
	y = (2 * y - 1)*p_t*z / p_n;
	return vec3(x, y, -z);
}

void shading_normal() {
	FragColor = vec4(texture(normalDTex, texCoord).xyz, 1.0);
}

void shading_fresnel_scale() {
	vec3 n = texture(normalDTex, texCoord).xyz;
	vec3 e = normalize(-getPos());
	float r = r0 + (1 - r0)*pow(1 - dot(n, e), 5);
	FragColor = vec4(r, r, r, 1.0);
}

void shading_fresnel() {
	vec3 n = texture(normalDTex, texCoord).xyz;
	vec3 e = normalize(-getPos());
	float r = r0 + (1 - r0)*pow(1 - dot(n, e), 5);
	
	vec3 view_ref = -e + 2 * n * dot(n, e);
	vec3 world_ref = mat3(iview) * view_ref;
	FragColor = texture(skyTex, world_ref);
	// FragColor = vec4(normalize(world_ref), 1);
	// FragColor = vec4(n, 1);
	// FragColor = vec4(abs(getPos()), 1);
}

void main() {

	// ze to z_ndc to gl_FragDepth
	// REF: https://computergraphics.stackexchange.com/questions/6308/why-does-this-gl-fragdepth-calculation-work?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	float ze = texture(zTex, texCoord).x;
	float z_ndc = proj(-ze);
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * z_ndc + gl_DepthRange.far + gl_DepthRange.near);	

	shading_fresnel();
}