#define STRINGIFY(s) #s

/* Stringify does not support # and newline, so hack outside out it */
const char* vshader = "# version 330 core\n" STRINGIFY(

	layout (location = 0) in vec3 aPos;

	uniform mat4 proj; 
	uniform mat4 view; 

//	out vec4 FragPos;	

	void main() {
		vec4 FragPos = view * vec4(aPos, 1.0);
		gl_Position = proj * FragPos;
	}
);

const char* fshader = "# version 330 core\n" STRINGIFY(

out vec4 FragColor;

void main() {
	FragColor = vec4(1., 1., 1., 1.);
}

);