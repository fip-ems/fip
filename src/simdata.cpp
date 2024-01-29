#include "simdata.h"
#include "yaml-cpp/yaml.h"
#include "utils.h"
#include "date.h"
#include "enumhelper.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <iostream>

namespace SimData
{
	std::string name;
	size_t W, H;
	size_t pitch;
	float dx, dt;
	bool is_variable_dt;
	Solver solver;
	double target_duration;
	double duration = 0;
	int iteration = 0;
	float* d_h1, * d_h2;
	float* d_qx1, * d_qx2, * d_qy1, * d_qy2;
	float* d_sohle;
	float* d_rei{ nullptr };
	float* d_hmax;
	float kSt_fixed{ 30.f };
	float* d_retention;
	float* h_sohle, * h_precip;
	float invalid_terrain = -9999.f;
	uint64_t* d_mask;
	float* dt_cfl_mins;
	size_t num_compute_blocks;

	bool is_staggered;
	std::vector<BoundaryCondition*> boundary_conditions;
	std::vector<double> save_state_times;

	namespace Sampling
	{
		int interval{ 0 };
		double timer{ duration };
		std::vector<SamplePoint> points;
	}
}

using namespace SimData;

ENUM_DATA(RandType, "z", "q", "close", "open");
ENUM_DATA(RandSide, "left", "right", "top", "bottom");
ENUM_DATA(Solver, "rmg", "cn");
ENUM_DATA(SampleType, "z", "h", "qx", "qy");


void initDeviceBuffers(const float* h_ah, const float* h_qx, const float* h_qy, const float* h_rei, const float* h_sohle)
{
	// gridDim.x * num_warps_per_tb * gridDim.y
	SimData::num_compute_blocks = (W / (num_block_cols * num_warps_per_tb) + 1) * num_warps_per_tb * (H / num_block_rows + 1);
	size_t num_bytes_line = W * sizeof(float);
	size_t num_bytes_mask = num_compute_blocks * sizeof(uint64_t);
	size_t num_bytes_timesteps = num_compute_blocks * sizeof(float);

	// allocate device memory
	checkCudaErrors(cudaMallocPitch(&d_h1, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_h2, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_hmax, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_qx1, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_qx2, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_qy1, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_qy2, &pitch, num_bytes_line, H + 3));
	checkCudaErrors(cudaMallocPitch(&d_sohle, &pitch, num_bytes_line, H + 3));
	if (h_rei) checkCudaErrors(cudaMallocPitch(&d_rei, &pitch, num_bytes_line, H + 3));

	checkCudaErrors(cudaMallocPitch(&d_retention, &pitch, num_bytes_line, H + 3));

	checkCudaErrors(cudaMallocHost(&dt_cfl_mins, num_bytes_timesteps));

	// set mask to entirely wet (all bits 1)
	checkCudaErrors(cudaMalloc(&d_mask, num_bytes_mask));
	checkCudaErrors(cudaMemset(d_mask, 0xFF, num_bytes_mask));

	// copy host to device memory
	checkCudaErrors(cudaMemcpy2D(d_h1, pitch, h_ah, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_h2, pitch, h_ah, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_hmax, pitch, h_ah, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_qx1, pitch, h_qx, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_qx2, pitch, h_qx, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_qy1, pitch, h_qy, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_qy2, pitch, h_qy, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_sohle, pitch, h_sohle, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	if (h_rei) checkCudaErrors(cudaMemcpy2D(d_rei, pitch, h_rei, num_bytes_line, num_bytes_line, H, cudaMemcpyHostToDevice));
	float* h_timesteps = new float[num_bytes_timesteps];
	for (int i = 0; i < num_bytes_timesteps; i++)
		h_timesteps[i] = 1.f;
	checkCudaErrors(cudaMemcpy(dt_cfl_mins, h_timesteps, num_bytes_timesteps, cudaMemcpyHostToDevice));
	delete[] h_timesteps;
}

BoundaryCondition* createBoundaryCondition(const YAML::Node& bc_node)
{
	std::string typeStr = bc_node["type"].as<std::string>();
	std::string sideStr = bc_node["side"].as<std::string>();
	bool isTimeseries = bc_node["q"].IsDefined() && bc_node["q"].IsSequence() || bc_node["z"].IsDefined() && bc_node["z"].IsSequence();

	auto type = stringToEnum<RandType>(typeStr);
	auto side = stringToEnum<RandSide>(sideStr);
	auto from = bc_node["from"].as<int>(0);
	auto to = bc_node["to"].as<int>(
		(side == RandSide::left || side == RandSide::right) ?
		(H - 1) :
		(W - 1));

	if (isTimeseries)
	{
		auto parseTimeSeries = [](std::string nodeName, const YAML::Node& node) {
			if (node.IsDefined())
				return TimeSeries::fromCsv(node[0].as<std::string>(), node[1].as<int>(), node[2].as<int>());
			return static_cast<TimeSeries*>(nullptr);
		};

		TimeSeries* q_timeseries = parseTimeSeries("q", bc_node["q"]);
		TimeSeries* z_timeseries = parseTimeSeries("z", bc_node["z"]);

		return new BoundaryConditionTimeseries(type, side, from, to, z_timeseries, q_timeseries);
	}
	else
	{
		auto q = bc_node["q"].as<float>(0.f);
		auto z = bc_node["z"].as<float>(0.f);
		return new BoundaryCondition(type, side, from, to, z, q);
	}
}

SamplePoint createSamplePoint(const YAML::Node& node)
{
	SamplePoint sp;
	sp.x = node["x"].as<int>();
	sp.y = node["y"].as<int>();
	sp.label = node["label"].as<std::string>();
	sp.type = stringToEnum<SampleType>(node["type"].as<std::string>());
	return sp;
}

void loadSimDataFileWithOffset(float* dst, const YAML::Node& parentNode, const std::string& nodeName)
{
	auto& node = parentNode[nodeName];
	auto filePath = node[0].as<std::string>();
	auto offset = node[1].as<int>();
	size_t gridSize = W * H;
	utils::loadFileIntoArray(filePath, dst, gridSize * sizeof(float), offset);
}

bool validateSimDataConfig(const YAML::Node& config)
{
	bool ok = true;

	auto error = [&ok](const YAML::Node& node, const std::string& msg) {
		ok = false;
		std::cerr << "Config error at line " << node.Mark().line + 1 << ": " << msg << std::endl;
	};
	auto assertInConfig = [&error](const YAML::Node& node, const std::string& a) {
		if (!node[a].IsDefined())
			error(node, "\"" + a + "\" missing");
	};
	auto assertXorInConfig = [&error](const YAML::Node& node, const std::string& a, const std::string& b) {
		if (node[a].IsDefined() && node[b].IsDefined())
			error(node[a], "Both \"" + a + "\" and \"" + b + "\" defined");
	};
	auto assertGoodPath = [&error](const YAML::Node& node, const std::string& a) {
		if (node[a].IsDefined() && (!node[a].IsSequence() || node[a].size() != 2))
			error(node[a], "file path \"" + a + "\" needs format [path, byte_offset]");
	};
	auto assertGoodTimeSeries = [&error](const YAML::Node& node, std::string a) {
		if (node[a].IsDefined() && (!node[a].IsSequence() || node[a].size() != 3))
			error(node[a], "timeseries \"" + a + "\" needs format [csv_filename, int_time_column, int_data_column]");
	};

	assertInConfig(config, "name");
	assertInConfig(config, "terrain");
	assertInConfig(config, "W");
	assertInConfig(config, "H");
	assertInConfig(config, "dt");
	assertInConfig(config, "dx");
	assertInConfig(config, "duration");
	assertXorInConfig(config, "kSt_var", "kSt");

	assertGoodPath(config, "terrain");
	assertGoodPath(config, "z");
	assertGoodPath(config, "qx");
	assertGoodPath(config, "qy");
	assertGoodPath(config, "kSt_var");

	if (!config["kSt_var"].IsDefined() && !config["kSt"].IsDefined())
		std::cerr << "Warning: both \"kSt_var\" and \"kSt\" missing, using default kSt " << SimData::kSt_fixed << std::endl;

	for (auto bc_node : config["boundary_conditions"])
	{
		assertInConfig(bc_node, "side");
		assertInConfig(bc_node, "type");
		std::string typeStr = bc_node["type"].as<std::string>();
		auto type = stringToEnum<RandType>(typeStr);
		if (type == RandType::z)
			assertInConfig(bc_node, "z");
		if (type == RandType::q) {
			assertInConfig(bc_node, "z");
			assertInConfig(bc_node, "q");
		}
		if (bc_node["z"].IsDefined() && bc_node["z"].IsSequence())
			assertGoodTimeSeries(bc_node, "z");
		if (bc_node["q"].IsDefined() && bc_node["q"].IsSequence())
			assertGoodTimeSeries(bc_node, "q");
	}

	if (config["sampling_interval"].IsDefined())
	{
		if (!config["sampling"].IsDefined() || !config["sampling"].IsSequence())
		{
			ok = false;
			std::cerr << "Configuration error: sampling interval specified but no sampling points given" << std::endl;
		}

		for (auto sampling_node : config["sampling"])
		{
			assertInConfig(sampling_node, "x");
			assertInConfig(sampling_node, "y");
			assertInConfig(sampling_node, "label");
			assertInConfig(sampling_node, "type");
		}
	}

	if (!config["save_state"].IsDefined())
	{
		std::cerr << "Warning: no save_state times given" << std::endl;
	}

	return ok;
}

void SimData::loadSimData(const std::string& fileName)
{
	YAML::Node config = YAML::LoadFile(fileName);

	if (!validateSimDataConfig(config))
		throw std::invalid_argument("one or more configuration errors found");

	SimData::name = config["name"].as<std::string>();
	SimData::target_duration = utils::solveSimpleMathExpression(config["duration"].as<std::string>());
	SimData::W = config["W"].as<size_t>();
	SimData::H = config["H"].as<size_t>();
	SimData::dt = config["dt"].as<float>();
	SimData::dx = config["dx"].as<float>();
	SimData::is_variable_dt = config["variable_dt"].as<bool>(true);
	SimData::solver = config["solver"].IsDefined() ? stringToEnum<Solver>(config["solver"].as<std::string>()) : Solver::rmg;
	SimData::kSt_fixed = config["kSt"].as<float>(kSt_fixed);
	SimData::is_staggered = solver == Solver::rmg;

	size_t gridSize = W * H;
	h_sohle = new float[gridSize];
	float* h_ah = new float[gridSize] {};
	float* h_qx = new float[gridSize] {};
	float* h_qy = new float[gridSize] {};
	float* h_rei{ nullptr };

	loadSimDataFileWithOffset(h_sohle, config, "terrain");
	if (config["no_data"].IsDefined())
		SimData::invalid_terrain = config["no_data"].as<float>();
	if (config["z"].IsDefined())
	{
		loadSimDataFileWithOffset(h_ah, config, "z");
		for (int i = 0; i < gridSize; i++) h_ah[i] = h_ah[i] - h_sohle[i];
	}
	if (config["qx"].IsDefined())
		loadSimDataFileWithOffset(h_qx, config, "qx");
	if (config["qy"].IsDefined())
		loadSimDataFileWithOffset(h_qy, config, "qy");
	if (config["kSt_var"].IsDefined())
	{
		h_rei = new float[gridSize];
		loadSimDataFileWithOffset(h_rei, config, "kSt_var");
	}

	for (auto bc_node : config["boundary_conditions"])
	{
		BoundaryCondition* bc = createBoundaryCondition(bc_node);
		boundary_conditions.push_back(bc);
	}


	initDeviceBuffers(h_ah, h_qx, h_qy, h_rei, h_sohle);

	if (h_ah) delete[] h_ah;
	if (h_qx) delete[] h_qx;
	if (h_qy) delete[] h_qy;
	if (h_rei) delete[] h_rei;



	if (config["sampling_interval"].IsDefined())
	{
		Sampling::interval = config["sampling_interval"].as<int>();
		for (auto sampling_node : config["sampling"])
			Sampling::points.push_back(createSamplePoint(sampling_node));
	}

	if (config["save_state"].IsDefined())
	{
		for (auto save_state_node : config["save_state"])
			save_state_times.push_back(utils::solveSimpleMathExpression(save_state_node.as<std::string>()));
		std::sort(save_state_times.begin(), save_state_times.end());
	}
}
