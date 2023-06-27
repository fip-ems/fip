#pragma once 
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

enum class VisType
{
	Mesh, Points, Wireframe
};

class Visualization
{
public:
	Visualization(VisType visType, std::string terrainTexturePath = "");

	void renderUntilExit(std::function<void()> launch_kernel);

	void addPlot(const std::vector<std::pair<float, float>>* data,
		std::string title, std::string xlabel = "x", std::string ylabel = "y");

private:
	void initGL(std::string terrainTexturePath);
	void initImgui();
	bool handleInput(float deltaTime);

	int iteration;
	VisType visType;

	// camera variables
	int camera_mode = 0;
	float viewMat[16];
	float cam_pos[3] = { 0.f, 2888.f, 0.f };
	float cam_look[3] = { 0.f, -0.9999999f, -0.0000003f };
	float cam_pivot[3] = { 0.f, 0.f, 0.f };
	const float up[3] = { 0, 1, 0 };

	// opengl variables
	GLFWwindow* window;
	GLuint shaderProgram;
	GLuint vbo, ibo;
	GLuint texWater, texTerrain, texQx, texQy, texColor;
	size_t vboSize;
	size_t iboSize;

	// CUDA graphics resources for opengl interop
	struct cudaGraphicsResource* texResZ;
	struct cudaGraphicsResource* texResQx;
	struct cudaGraphicsResource* texResQy;

	struct Plot
	{
		const std::vector<std::pair<float, float>>* data;
		std::string title;
		std::string xlabel;
		std::string ylabel;
	};
	std::vector<Plot> plots;
};
