#include <string.h>
#include <iostream>
#include <vector>
#include <thread>

#include "simdata.h"
#include "vis.h"
#include "shaders.h"
#define MATH_3D_IMPLEMENTATION
#include "math_3d.h"
#include "helper_cuda.h"
#include "cuda.h"
#include "orbit_camera.h"
#define FLYTHROUGH_CAMERA_IMPLEMENTATION
#include "flythrough_camera.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot/implot.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// input related variables
bool do_animate_single_step = false;
bool did_just_resize = false;
double mouse_pressed_x;
double mouse_pressed_y;
double scroll_offset = 0;

// forward declarations for glfw callbacks
void framebuffer_size_callback(GLFWwindow*, int, int);
void mouse_button_callback(GLFWwindow*, int, int, int);
void scroll_callback(GLFWwindow*, double, double);
void key_callback(GLFWwindow*, int, int, int, int);


Visualization::Visualization(VisType visType, std::string terrainTexturePath)
{
	this->iteration = 0;
	this->visType = visType;

	initGL(terrainTexturePath);
	initImgui();
}

void Visualization::initImgui()
{
	// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
	// GL ES 2.0 + GLSL 100
	const char* glsl_version = "#version 100";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif


	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	ImPlot::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
}

void Visualization::initGL(std::string terrainTexturePath)
{
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(512, 512, "FIP", NULL, NULL);
	if (window == NULL)
	{
		std::cerr << "Failed to create GLFW window" << std::endl;
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		exit(EXIT_FAILURE);
	}

	glViewport(0, 0, 512, 512);

	// compile shaders and link program
	GLuint vertId = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragId = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vertId, 1, &vertShader, 0);
	glCompileShader(vertId);
	glShaderSource(fragId, 1, &fragShader, 0);
	glCompileShader(fragId);
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertId);
	glAttachShader(shaderProgram, fragId);
	glLinkProgram(shaderProgram);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(shaderProgram, 256, 0, temp);
		std::cerr << "Failed to link program:\n" << temp << std::endl;
		exit(EXIT_FAILURE);
	}

	// set uniforms
	glUseProgram(shaderProgram);
	//mat4_t projection = m4_ortho(-1, 1, -1, 1, -100, 100000);
	mat4_t projection = m4_perspective(80.f, 1.0f, 0.01f, 10000.f);
	// move grid center to origin
	mat4_t model = m4_translation({ -0.5f * SimData::W, 0.f, -0.5f * SimData::H });
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model.m00);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m00);
	glUniform1i(glGetUniformLocation(shaderProgram, "W"), SimData::W);
	glUniform1i(glGetUniformLocation(shaderProgram, "H"), SimData::H);
	glUniform1i(glGetUniformLocation(shaderProgram, "texWater"), 0);
	glUniform1i(glGetUniformLocation(shaderProgram, "texTerrain"), 1);
	glUniform1i(glGetUniformLocation(shaderProgram, "texQx"), 2);
	glUniform1i(glGetUniformLocation(shaderProgram, "texQy"), 3);
	glUniform1i(glGetUniformLocation(shaderProgram, "texColor"), 4); 
	glUniform1f(glGetUniformLocation(shaderProgram, "terrainScale"), 1.f);
	glUniform1f(glGetUniformLocation(shaderProgram, "invalidTerrain"), SimData::invalid_terrain);

	// generate vertex buffer object and index buffer for trianglestrip
	int dx = 5;
	const int vboWidth = std::ceil(SimData::W / float(dx)) + 1;
	const int vboHeight = std::ceil(SimData::H / float(dx)) + 1;
	std::vector<int> points;
	std::vector<unsigned int> indices;
	for (int i = 0; i < vboHeight; i++)
	{
		for (int j = 0; j < vboWidth; j++)
		{
			points.push_back(std::min(j * dx, (int)SimData::W));
			points.push_back(std::min(i * dx, (int)SimData::H));

			if (i < vboHeight-1)
			{
				indices.push_back(i * vboWidth + j);
				indices.push_back((i + 1) * vboWidth + j);
			}
		}
		if (i < vboHeight-1)
		{
			// degenerate triangles
			indices.push_back((i + 2) * vboWidth - 1);
			indices.push_back((i + 1) * vboWidth);
		}
	}
	vboSize = points.size() / 2;
	iboSize = indices.size();
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(unsigned int), points.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_INT, GL_FALSE, 2 * sizeof(int), 0);
	// generate index buffer object
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	// generate texture with one channel and type float.
	// represents height of grid at texture coordinate (u,v)
	int textureCount = 0;
	for (auto tex : { &texWater, &texTerrain, &texQx, &texQy })
	{
		glActiveTexture(GL_TEXTURE0 + textureCount++);
		glGenTextures(1, tex);
		glBindTexture(GL_TEXTURE_2D, *tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SimData::W, SimData::H, 0, GL_RED, GL_FLOAT, SimData::h_sohle);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	// generate RGB texture for satellite / aerial image
	if (!terrainTexturePath.empty())
	{
		glUniform1i(glGetUniformLocation(shaderProgram, "hasTexture"), 1);
		int x, y, comp;
		stbi_set_flip_vertically_on_load(true);
		auto image = stbi_load(terrainTexturePath.c_str(), &x, &y, &comp, 3);
		glActiveTexture(GL_TEXTURE0 + textureCount++);
		glGenTextures(1, &texColor);
		glBindTexture(GL_TEXTURE_2D, texColor);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		stbi_image_free(image);
	}

	// register textures with cuda
	checkCudaErrors(cudaGraphicsGLRegisterImage(&texResZ, texWater, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	checkCudaErrors(cudaGraphicsGLRegisterImage(&texResQx, texQx, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	checkCudaErrors(cudaGraphicsGLRegisterImage(&texResQy, texQy, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}


void Visualization::renderUntilExit(std::function<void()> launch_kernel)
{
	size_t num_bytes_line = SimData::W * sizeof(float);

	// map texture for writing from CUDA
	// Accessing a resource through OpenGL, Direct3D, or another CUDA context while it is mapped produces undefined results.
	cudaArray_t mappedArrayZ, mappedArrayQx, mappedArrayQy;
	checkCudaErrors(cudaGraphicsMapResources(3, &texResZ));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&mappedArrayZ, texResZ, 0, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&mappedArrayQx, texResQx, 0, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&mappedArrayQy, texResQy, 0, 0));
	//checkCudaErrors(cudaGraphicsUnmapResources(1, &tex_res, 0));

	cam_pos[0] = 0.f;
	cam_pos[1] = fminf(3000, fmaxf(SimData::W, SimData::H));
	cam_pos[2] = 0.f;

	flythrough_camera_update(cam_pos, cam_look, up, viewMat,
		0.016f, 0, 0, 90.f, 0, 0, 0, 0, 0, 0, 0, 0,	0);

	// Imgui state
	float terrainScale = 1.f;
	int vis = 0;
	int simSpeed = 10;
	bool animating = false;
	bool haveNewAnimationData = true;
	float maxWaterDepth = 5;
	float maxWaterVelo = 5;

	double lastFrameTime = glfwGetTime();
	double lastCopyTime = 0;

	CUcontext cudaContext;
	cuCtxGetCurrent(&cudaContext);
	std::thread t([&]() {
		cuCtxSetCurrent(cudaContext);
		while (!glfwWindowShouldClose(window))
		{
			// launch CUDA kernel
			for (int i = 0; i < simSpeed; i++)
			{
				if (animating || do_animate_single_step)
				{
					iteration++;
					do_animate_single_step = false;
					haveNewAnimationData = true;
					launch_kernel();
					cudaDeviceSynchronize();
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
	});

	GLuint uniformView = glGetUniformLocation(shaderProgram, "view");
	GLuint uniformProjection = glGetUniformLocation(shaderProgram, "projection");
	GLuint uniformViewPos = glGetUniformLocation(shaderProgram, "viewPos");
	GLuint uniformVis = glGetUniformLocation(shaderProgram, "vis");
	GLuint uniformTerrainScale = glGetUniformLocation(shaderProgram, "terrainScale");
	GLuint uniformHighlight = glGetUniformLocation(shaderProgram, "highlight_xy");
	GLuint uniformWaterDepth = glGetUniformLocation(shaderProgram, "maxDepth");
	GLuint uniformWaterVelo = glGetUniformLocation(shaderProgram, "maxVelocity");

	glUniform1f(uniformWaterDepth, maxWaterDepth);
	glUniform1f(uniformWaterVelo, maxWaterVelo);

	while (!glfwWindowShouldClose(window))
	{
		glDisable(GL_SCISSOR_TEST);
		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		// change winding order such that degenerate triangles are culled correctly in wireframe mode
		glFrontFace(GL_CW);
		glPolygonMode(GL_FRONT_AND_BACK, visType == VisType::Wireframe ? GL_LINE : GL_FILL);
		glUseProgram(shaderProgram);
		glBindVertexArray(0);
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		mat4_t projection = m4_perspective(80.f, float(display_w) / float(display_h), 0.01f, 10000.f);
		glUniformMatrix4fv(uniformProjection, 1, GL_FALSE, &projection.m00);

		double timeNow = glfwGetTime();
		float deltaTime = timeNow - lastFrameTime;
		lastFrameTime = timeNow;

		// do not render when idle (not using menu, not simulating, not resizing window, not moving camera)
		bool doRender = haveNewAnimationData | ImGui::GetIO().WantCaptureMouse | did_just_resize;
		did_just_resize = false;
		
		// move camera
		if (!ImGui::GetIO().WantCaptureMouse)
			doRender |= handleInput(deltaTime);
		glUniformMatrix4fv(uniformView, 1, GL_FALSE, viewMat);
		glUniform3fv(uniformViewPos, 1, &cam_pos[0]);

		// copy same device memory to texture, to sample it in fragment shader
		if (timeNow - lastCopyTime >= 0.04 && haveNewAnimationData)
		{
			haveNewAnimationData = false;
			lastCopyTime = timeNow;
			checkCudaErrors(cudaMemcpy2DToArrayAsync(mappedArrayZ, 0, 0, SimData::d_h1, SimData::pitch, num_bytes_line, SimData::H, cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy2DToArrayAsync(mappedArrayQx, 0, 0, SimData::d_qx1, SimData::pitch, num_bytes_line, SimData::H, cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy2DToArrayAsync(mappedArrayQy, 0, 0, SimData::d_qy1, SimData::pitch, num_bytes_line, SimData::H, cudaMemcpyDeviceToDevice));
		}

		// render grid
		if (doRender)
		{
			// setup render state
			glClearColor(0.f, 0.5f, 1.0f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);


			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			if (visType == VisType::Mesh || visType == VisType::Wireframe)
				glDrawElements(GL_TRIANGLE_STRIP, iboSize, GL_UNSIGNED_INT, (void*)0);
			else
				glDrawArrays(GL_POINTS, 0, vboSize);
			// unbind
			glBindBuffer(GL_ARRAY_BUFFER, 0);


			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			{
				ImGui::Begin("FIP");
				if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
				{
					if (animating && ImGui::Button("Stop Simulation"))
						animating = false;
					else if (!animating && ImGui::Button("Start Simulation"))
						animating = true;
					ImGui::Text("Simulation Duration: %.2f s", SimData::duration);
					int cfl_block_x = int(SimData::h_cfl_ts[1]) % (SimData::W / (28 * 4) * 4 + 4);
					int cfl_block_y = int(SimData::h_cfl_ts[1]) / (SimData::W / (28 * 4) * 4 + 4);
					glUniform2i(uniformHighlight, cfl_block_x * 28, cfl_block_y * 50);
					ImGui::Text("Timestep: %.3f (%.3f, [%d, %d])", SimData::dt, SimData::h_cfl_ts[0], cfl_block_x, cfl_block_y);
					ImGui::SliderInt("Simulation Speed", &simSpeed, 1, 200);
				}
				ImGui::Separator();
				if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
				{
					ImGui::Text("Visualize: "); ImGui::SameLine();
					if (ImGui::RadioButton("Water depth", &vis, 0))
						glUniform1i(uniformVis, 0);
					ImGui::SameLine();
					if (ImGui::RadioButton("Velocity", &vis, 1))
						glUniform1i(uniformVis, 1);
					if (ImGui::SliderFloat("Maximum Depth", &maxWaterDepth, 0.f, 15.f, "%.2f m", ImGuiSliderFlags_Logarithmic))
						glUniform1f(uniformWaterDepth, maxWaterDepth);
					if (ImGui::SliderFloat("Maximum Velocity", &maxWaterVelo, 0.f, 15.f, "%.2f m/s", ImGuiSliderFlags_Logarithmic))
						glUniform1f(uniformWaterVelo, maxWaterVelo);
					ImGui::Text("Render as: "); ImGui::SameLine();
					ImGui::RadioButton("Mesh", (int*)&visType, 0); ImGui::SameLine();
					ImGui::RadioButton("Points", (int*)&visType, 1); ImGui::SameLine();
					ImGui::RadioButton("Wireframe", (int*)&visType, 2);
					if (ImGui::SliderFloat("Terrain Scale", &terrainScale, 0.f, 5.f, "%.1f"))
						glUniform1f(uniformTerrainScale, terrainScale);
					ImGui::Text("Camera mode:"); ImGui::SameLine();
					ImGui::RadioButton("Orbit", &camera_mode, 0); ImGui::SameLine();
					ImGui::RadioButton("First Person", &camera_mode, 1);
				}
				ImGui::End();
				for (auto& plot : plots)
				{
					ImGui::Begin(plot.title.c_str());
					if (ImPlot::BeginPlot(plot.title.c_str(), { -1, -1 }))
					{
						ImPlot::SetupAxes(plot.xlabel.c_str(), plot.ylabel.c_str());
						ImPlot::SetupAxesLimits(0, 1000, -1, 1, ImPlotCond_Once);
						auto limits = ImPlot::GetPlotLimits();
						//ImPlot::SetupAxes(plot.xlabel.c_str(), plot.ylabel.c_str());
						ImPlot::PlotLine<float>("", &plot.data->data()->first, &plot.data->data()->second, plot.data->size(),
							0, 0, 8);
						ImPlot::EndPlot();
						if (plot.data->size() > 0)
						{
							float lastx = plot.data->rbegin()->first;
							float lasty = plot.data->rbegin()->second;
							if (lasty < limits.Y.Min)
								ImPlot::SetNextAxesLimits(limits.X.Min, limits.X.Max, lasty * 1.5, limits.Y.Max, ImPlotCond_Always);
							if (lastx > limits.X.Max)
								ImPlot::SetNextAxesLimits(limits.X.Min, lastx * 2.0, limits.Y.Min, limits.Y.Max, ImPlotCond_Always);
							if (lasty > limits.Y.Max)
								ImPlot::SetNextAxesLimits(limits.X.Min, limits.X.Max, limits.Y.Min, lasty * 1.5, ImPlotCond_Always);
						}
					}
					ImGui::End();
				}
			}

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(window);
		}
		else
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}

		glfwPollEvents();
	}

	t.join();

	checkCudaErrors(cudaGraphicsUnmapResources(1, &texResZ, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &texResQx, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &texResQy, 0));

	// Imgui Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	glfwTerminate();
}


void Visualization::addPlot(const std::vector<std::pair<float, float>>* data,
	std::string title, std::string xlabel, std::string ylabel)
{
	plots.push_back({ data, title, xlabel, ylabel });
}

bool Visualization::handleInput(float deltaTime)
{
	bool forw = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
	bool left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
	bool back = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
	bool right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
	int fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
	int slow = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
	bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS;
	bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS;

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	bool view_changed = lmb || rmb || scroll_offset;

	if (camera_mode == 0) 
	{
		orbitcam_update(
			cam_pos, cam_pivot, up, cam_look, viewMat,
			deltaTime,
			1.f, 0.2f, xpos - mouse_pressed_x, ypos - mouse_pressed_y, scroll_offset, lmb, rmb);

		mouse_pressed_x = xpos;
		mouse_pressed_y = ypos;
	}
	else if (camera_mode == 1 && lmb)
	{
		flythrough_camera_update(
			cam_pos, cam_look, up, viewMat,
			deltaTime,
			100.f + fast * 300.f - slow * 75.f,
			0.2f,
			89.9f,
			xpos - mouse_pressed_x, ypos - mouse_pressed_y,
			forw, left, back, right,
			0,
			0,
			0
		);
		glfwSetCursorPos(window, mouse_pressed_x, mouse_pressed_y);
	}

	scroll_offset = 0.;

	return view_changed;
}

// glfw callbacks

void framebuffer_size_callback(GLFWwindow*, int width, int height)
{
	glViewport(0, 0, width, height);
	did_just_resize = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		glfwGetCursorPos(window, &mouse_pressed_x, &mouse_pressed_y);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else if (action == GLFW_RELEASE)
	{
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	scroll_offset += yoffset;
}

void key_callback(GLFWwindow*, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_RIGHT && (action == GLFW_REPEAT || action == GLFW_PRESS))
	{
		do_animate_single_step = true;
	}
}
