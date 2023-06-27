#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include "kernels.h"
#include "simdata.h"
#include "bc_kernel_gen.h"
#include "date.h"
#include "helper_cuda.h"

#ifndef NO_VIS
#include "vis.h"
#endif

using namespace SimData;

void* boundaryConditionKernelLauncher{ nullptr };

std::vector<std::pair<float, float>> volumePlotData;

void dumpDeviceBuffers();
float* dumpDeviceBuffer(float* device_buf);


void controlTimestep()
{
	float targetTimestep = h_cfl_ts[0] * .5f;
	constexpr float dc = 0.001f;
	if (dt < targetTimestep && targetTimestep - dt > dc)
	{
		dt *= 1.001f;
	}
	else if (dt > targetTimestep && dt - targetTimestep > dc)
	{
		dt *= 0.99f;
	}
}

void sampleHeader()
{
	std::cout << "t;";
	for (SamplePoint& sp : Sampling::points)
		std::cout << sp.label << ";";
	std::cout << std::endl;
}

void sample()
{
	cudaDeviceSynchronize();
	std::cout << duration << ";";
	for (SamplePoint& sp : Sampling::points)
	{
		float val;
		int d_idx = sp.y * (pitch / 4) + sp.x;
		int h_idx = sp.y * W + sp.x;
		if (sp.type == SampleType::z)
		{
			cudaMemcpy(&val, d_h1 + d_idx, sizeof(float), cudaMemcpyDeviceToHost);
			float sohle = h_sohle[h_idx];
			val += sohle;
		}
		if (sp.type == SampleType::h)
		{
			cudaMemcpy(&val, d_h1 + d_idx, sizeof(float), cudaMemcpyDeviceToHost);
		}
		if (sp.type == SampleType::qx)
		{
			cudaMemcpy(&val, d_qx1 + d_idx, sizeof(float), cudaMemcpyDeviceToHost);
		}
		if (sp.type == SampleType::qy)
		{
			cudaMemcpy(&val, d_qy1 + d_idx, sizeof(float), cudaMemcpyDeviceToHost);
		}
		std::cout << val << ";";
	}
	std::cout << std::endl;
}

void launchBoundaryConditionKernel()
{
	std::vector<JitParam> params;
	for (int i = 0; i < boundary_conditions.size(); i++)
	{
		auto bc = boundary_conditions[i];
		params.push_back(bc->getQ(duration));
		params.push_back(bc->getZ(duration));
		params.push_back(bc->getFrom());
		params.push_back(bc->getTo());
	}

	BoundaryConditionKernel::launch(boundaryConditionKernelLauncher, params);
}

void simulate()
{
	if (is_variable_dt && iteration % 10 == 9)
	{
		launchTimestepReduceKernel();
		cudaDeviceSynchronize();
		controlTimestep();
	}

	if (duration >= Sampling::timer && Sampling::interval > 0)
	{
		Sampling::timer += Sampling::interval;
		sample();
	}

	if (iteration == 0)
		launchBoundaryConditionKernel();

	launchComputeKernel();

	iteration++;
	std::swap(d_qx1, d_qx2);
	std::swap(d_qy1, d_qy2);
	std::swap(d_h1, d_h2);

	launchBoundaryConditionKernel();

	if (iteration % 100 == 0)
	{
		launchFloodPlainKernel();
	}

	if (save_state_times.size() > 0 && duration >= save_state_times.front())
	{
		save_state_times.erase(save_state_times.begin());
		dumpDeviceBuffers();
	}
}

void simulateVis()
{
	//if (iteration % 100 == 0)
	//{
	//	static double startVolume = -1;
	//	auto h = dumpDeviceBuffer(d_h2);
	//	double volume = 0;
	//	for (int i = 0; i < W * H; i++)
	//		volume += h[i];
	//	if (startVolume == -1) startVolume = volume;
	//	volumePlotData.push_back({ duration, volume - startVolume });
	//	delete[] h;
	//}

	simulate();
}

void save2DArrayToFile(const std::string& fileName, const float* arr, size_t width, size_t height)
{
	std::ofstream file;
	file.open(fileName, std::ios::binary);

	if (!file)
		throw std::invalid_argument("Failed to open file " + fileName);

	file.write(reinterpret_cast<const char*>(&width), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(&height), sizeof(size_t));
	size_t gridSize = width * height;
	file.write(reinterpret_cast<const char*>(arr), gridSize * sizeof(float));
	file.close();
}


float* dumpDeviceBuffer(float* device_buf)
{
	checkCudaErrors(cudaDeviceSynchronize());

	size_t num_bytes_line = W * sizeof(float);
	float* h_u = new float[W * H];
	checkCudaErrors(cudaMemcpy2D(h_u, num_bytes_line, device_buf, pitch, num_bytes_line, H, cudaMemcpyDeviceToHost));
	return h_u;
}

void dumpDeviceBuffers()
{
	checkCudaErrors(cudaDeviceSynchronize());

	size_t num_bytes_line = W * sizeof(float);
	float* h_u = new float[W * H];

	auto timestamp = date::format("%Y-%m-%d-%H-%M", std::chrono::system_clock::now());

	checkCudaErrors(cudaMemcpy2D(h_u, num_bytes_line, d_h2, pitch, num_bytes_line, H, cudaMemcpyDeviceToHost));
	save2DArrayToFile(SimData::name +"_ah_" + std::to_string((int)duration) + "_" + timestamp, h_u, W, H);

	checkCudaErrors(cudaMemcpy2D(h_u, num_bytes_line, d_qx2, pitch, num_bytes_line, H, cudaMemcpyDeviceToHost));
	save2DArrayToFile(SimData::name + "_qx_" + std::to_string((int)duration) + "_" + timestamp, h_u, W, H);

	checkCudaErrors(cudaMemcpy2D(h_u, num_bytes_line, d_qy2, pitch, num_bytes_line, H, cudaMemcpyDeviceToHost));
	save2DArrayToFile(SimData::name + "_qy_" + std::to_string((int)duration) + "_" + timestamp, h_u, W, H);

	checkCudaErrors(cudaMemcpy2D(h_u, num_bytes_line, d_hmax, pitch, num_bytes_line, H, cudaMemcpyDeviceToHost));
	save2DArrayToFile(SimData::name + "_flood_plains_" + std::to_string((int)duration) + "_" + timestamp, h_u, W, H);

	delete[] h_u;
}

int main(int argc, char** argv)
{
	bool vis = true;

	std::string simDataFile = "inde.yml";
	if (argc < 2)
		std::cerr << "Warning: no config file given, using default " << simDataFile << std::endl;
	else
		simDataFile = argv[1];

	try
	{
		SimData::loadSimData(simDataFile);
	}
	catch (std::exception& ex)
	{
		std::cerr << "Config error: " << ex.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	boundaryConditionKernelLauncher = BoundaryConditionKernel::generate();
	if (Sampling::points.size() > 0)
		sampleHeader();
	
#ifndef NO_VIS
	if (vis)
	{
		Visualization vis{ VisType::Mesh, ""};
		//vis.addPlot(&volumePlotData, "Volume error", "time", "error");
		vis.renderUntilExit(&simulateVis);
	}
	else
#endif
	{
		double totalTime = 0.0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		while (duration < SimData::target_duration)
		{
			simulate();
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds;
		cudaEventElapsedTime(&milliseconds, start, stop);
		float seconds = milliseconds / 1000.f;

		std::cerr << std::dec << "Finished " << iteration << " iterations after " << seconds << " s (" << std::fixed << seconds / iteration << " s/it)" << std::endl;
	}

	cudaDeviceReset();

	// free host memory
	if (h_sohle)
		delete[] h_sohle;

	return 0;
}
