# PBF-CUDA
This project is an implementation of **Position Based Fluids[1]** written in CUDA and modified **Screen Space Fluids Rendering[2]**
written in OpenGL. 

## Gallery
**Double Dam Break 32K particles** 

<img src="https://github.com/naeioi/PBF-CUDA/raw/master/figs/double-dam.png" alt="Double Dam Break" width="700">

See [figs](https://github.com/naeioi/PBF-CUDA/tree/master/figs) for additonal screenshots of single dam, surface normal map and sweeping boundary scene.

## Features
**Techniques**
- Imcompressible fluids simulation using Macklin[1]. Vorticity confinement is not implemented. 
- Screen space surface reconstruction based on Laan[2]. 
- Surface smoothed by applying bilateral filter on depth texture.
- GPU particle neighbor searching using Green[3].
- GUI using [NanoGUI](https://github.com/wjakob/nanogui). Talk with [wjakob](https://github.com/wjakob) last year inspired me for this project. 

**Evaluation** 

Performance evaluated on a notebook with GTX 1050, i7-7700 HQ.

Under configuration of 4 incompressibility confinement iterations and 2 smoothing iterations each frame, 
this implementation runs at ~30 fps with 32K particles in Double Dam Break scene (Screenshots above).

## Dependencies
- CUDA 9.1
- Visual Studio 2017 15.4.5. \
  Higher version fails to build CUDA 9, as of 15.7.3. \
  See [Microsoft vcpkg Github issue](https://github.com/Microsoft/vcpkg/issues/2814). 
- GPU and driver support OpenGL 4.5

## Licence
[Apache License 2.0 (Apache-2.0)](https://www.apache.org/licenses/LICENSE-2.0)

## References
[1] M. Macklin and M. Müller, “Position based fluids,” ACM Trans. Graph., vol. 32, no. 4, p. 1, Jul. 2013. \
[2] W. J. van der Laan, S. Green, and M. Sainz, “Screen space fluid rendering with curvature flow,” in Proceedings of the 2009 symposium on Interactive 3D graphics and games - I3D ’09, 2009, p. 91. \
[3] S. Green, “Particle Simulation using CUDA,” cse.uaa.alaska.edu, no. September, pp. 1–12, 2013. \
[4] [Learn OpenGL, extensive tutorial resource for learning Modern OpenGL](https://learnopengl.com/) for great skybox tutorial. 
