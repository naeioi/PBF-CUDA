#define STRINGIFY(s) #s

const char* vshader = STRINGIFY(

	# version 330 core

	layout(location = 0) in vec3 aPos;

	uniform mat4 proj;
	uniform mat4 view;
	uniform mat4 obj;

//	out vec4 FragPos;	

	void main() {
		mat4 view_obj = view * obj;
		vec4 FragPos = view_obj * vec4(aPos, 1.0);
		gl_Position = proj * FragPos;
	}
);

const char* fshader = STRINGIFY(

# version 330 core

out vec4 FragColor;

void main() {
	FragColor = float3(1., 1., 1., 1);
}

);