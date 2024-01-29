#pragma once
#include <cstdint>
#include <chrono>
#include "boundary_condition.h"

enum class Solver
{
	rmg, cn
};

enum class SampleType
{
	z, h, qx, qy
};

struct SamplePoint
{
	int x, y;
	std::string label;
	SampleType type;
};

namespace SimData
{
	extern std::string name;
	extern size_t W, H;
	extern size_t pitch;
	extern float dx, dt;
	extern bool is_variable_dt;
	extern Solver solver;
	extern double target_duration;
	extern double duration;
	extern int iteration;
	extern float* d_h1, * d_h2;
	extern float* d_qx1, * d_qx2, * d_qy1, * d_qy2;
	extern float* d_sohle;
	extern float* d_rei, kSt_fixed;
	extern float* d_hmax;
	extern float* d_retention;
	extern float* h_sohle, * h_precip;
	extern float invalid_terrain;
	extern uint64_t* d_mask;
	extern float* dt_cfl_mins;
	extern size_t num_compute_blocks;
	extern const size_t num_warps_per_tb;
	extern const size_t num_block_rows;
	extern const size_t num_block_cols;
	constexpr float G = 9.81f;
	
	extern bool is_staggered;
	extern std::vector<BoundaryCondition*> boundary_conditions;
	extern std::vector<double> save_state_times;

	namespace Sampling
	{
		extern int interval;
		extern double timer;
		extern std::vector<SamplePoint> points;
	}
	
	extern void loadSimData(const std::string& fileName);
	extern void setupPrecipitationRadolan(float* h_precip, const std::string& fileName, int x_offset, int y_offset);
};
