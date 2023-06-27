
#define NOMINMAX
#include "bc_kernel_gen.h"
#include "simdata.h"
#include "utils.h"
//#define JITIFY_PRINT_INSTANTIATION 1
#include "jitify.hpp"

using namespace SimData;

const char* src_side_right = R"(
if (blockIdx.y == {0}) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + from{0};
	if (tid > to{0})
		return;
	int idx = tid * pitch + W - 1;
	int idx_staggered = idx - 1;
	int idx_neighbor = idx - 1;
	int idx_second_neighbor = idx - 2;
	int idx_neighbor_staggered = idx - 2;
	float* q = qx;
	float* q_lateral = qy;
	{1}
}
)";

const char* src_side_left = R"(
if (blockIdx.y == {0}) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + from{0};
	if (tid > to{0})
		return;
	int idx = tid * pitch;
	int idx_staggered = idx;
	int idx_neighbor = idx + 1;
	int idx_second_neighbor = idx + 2;
	int idx_neighbor_staggered = idx + 1;
	float* q = qx;
	float* q_lateral = qy;
	{1}
}
)";

const char* src_side_top = R"(
if (blockIdx.y == {0}) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + from{0};
	if (tid > to{0})
		return;
	int idx = tid + (H - 1) * pitch;
	int idx_staggered = idx - pitch;
	int idx_neighbor = idx - pitch;
	int idx_second_neighbor = idx - 2 * pitch;
	int idx_neighbor_staggered = idx - 2 * pitch;
	float* q = qy;
	float* q_lateral = qx;
	{1}
}
)";

const char* src_side_bottom = R"(
if (blockIdx.y == {0}) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + from{0};
	if (tid > to{0})
		return;
	int idx = tid;
	int idx_staggered = idx;
	int idx_neighbor = idx + pitch;
	int idx_second_neighbor = idx + 2 * pitch;
	int idx_neighbor_staggered = idx + pitch;
	float* q = qy;
	float* q_lateral = qx;
	{1}
}
)";

const char* src_type_z = R"(
	float ah = max(0.f, zval{0} - sohle[idx]);
	h[idx] = ah;
	q[idx] = 0.99f * (q[idx_neighbor] / max(h[idx_neighbor], 0.01f)) * ah;
)";

const char* src_type_z_staggered = R"(
	h[idx] = max(0.f, zval{0} - sohle[idx]);
	h[idx_neighbor] = max(0.f, zval{0} - sohle[idx_neighbor]);
	q[idx_staggered] = -0.f;
	//q[idx_staggered] = 0.95f * (q[idx_neighbor_staggered] / max(h[idx_neighbor], 0.01f)) * h[idx_neighbor];
)";

const char* src_type_q = R"(
	float ah = zval{0} - sohle[idx];
	if (ah < 0.f) ah = 0.f;
	q[idx] = sqrtf(G * ah) * ah * qval{0};

	if (ah > 0.f)
		h[idx] = max(ah/2, h[idx_neighbor] + sohle[idx_neighbor] - sohle[idx]);
)";

const char* src_type_q_staggered = R"(
	float ah = zval{0} - sohle[idx];
	if (ah < 0.f) ah = 0.f;
	q[idx_staggered] = sqrtf(G * ah) * ah * qval{0};
	if (ah > 0.f) {
		float z_neighbor = h[idx_neighbor] + sohle[idx_neighbor];
		h[idx] = max(ah/2, z_neighbor - sohle[idx]);
	}
)";

const char* src_type_close = R"(
	q[idx] = 0.f;
	q_lateral[idx] = q_lateral[idx_neighbor];
	h[idx] = h[idx_neighbor];
)";

const char* src_type_close_staggered = R"(
	q[idx_staggered] = 0.f;
	q_lateral[idx] = q_lateral[idx_neighbor];
	h[idx] = h[idx_neighbor];
)";

const char* src_type_open = R"(
	float ah_neighbor = h[idx_neighbor];
	float z_neighbor = ah_neighbor + sohle[idx_neighbor];
	float sign = idx - idx_neighbor;
	if (ah_neighbor > 0.f) {
		q[idx] = sign * sqrtf(G * ah_neighbor) * ah_neighbor;
		h[idx] = min(ah_neighbor, max(0.f, z_neighbor - sohle[idx]));
	}
)";

const char* src_type_open_staggered = R"(
	float ah_neighbor = h[idx_neighbor];
	float z_neighbor = ah_neighbor + sohle[idx_neighbor];
	float sign = idx - idx_neighbor;
	if (ah_neighbor > 0.f) {
		q[idx_staggered] = sign * sqrtf(G * ah_neighbor) * ah_neighbor;
		h[idx] = min(ah_neighbor, max(0.f, z_neighbor - sohle[idx]));
	}
)";

const char* src_params = R"(,
	float qval{0}, float zval{0}, int from{0}, int to{0}
)";

const char* src_randbed = R"(bc_program
constexpr float DH = 0.01f;
constexpr float G = 9.81f;

__global__
void randbed(float* h, float* qx, float* qy, float* sohle, int W, int H, int pitch{0})
{
	{1}
}
)";

void* BoundaryConditionKernel::generate()
{
	// generate source code based on given boundary conditions
	std::string src_all_conditions = "";
	std::string src_all_params = "";
	for (int i = 0; i < boundary_conditions.size(); i++)
	{
		auto bc = boundary_conditions[i];
		const char* src_type = "";
		const char* src_side = "";
		if (bc->getSide() == RandSide::left) src_side = src_side_left;
		if (bc->getSide() == RandSide::right) src_side = src_side_right;
		if (bc->getSide() == RandSide::top) src_side = src_side_top;
		if (bc->getSide() == RandSide::bottom) src_side = src_side_bottom;
		if (bc->getType() == RandType::z) src_type = is_staggered ? src_type_z_staggered : src_type_z;
		if (bc->getType() == RandType::q) src_type = is_staggered ? src_type_q_staggered : src_type_q;
		if (bc->getType() == RandType::open) src_type = is_staggered ? src_type_open_staggered : src_type_open;
		if (bc->getType() == RandType::close) src_type = is_staggered ? src_type_close_staggered : src_type_close;
		std::string src_fmt_params = utils::miniFormat(src_params, {std::to_string(i)});
		std::string src_fmt_type = utils::miniFormat(src_type, {std::to_string(i)});
		std::string src_fmt_side = utils::miniFormat(src_side, {std::to_string(i), src_fmt_type});
		src_all_params = src_all_params.append(src_fmt_params);
		src_all_conditions = src_all_conditions.append(src_fmt_side);
	}
	std::string src = utils::miniFormat(src_randbed, {src_all_params, src_all_conditions});
	
	// JIT compile the generated source code 
	static jitify::JitCache kernel_cache;
	dim3 tpb{ 64 };
	dim3 blocks(std::max(W, H) / 64 + 1, boundary_conditions.size());
	jitify::KernelLauncher* launcher = new jitify::KernelLauncher;
	
	*launcher = kernel_cache
		.program(src, 0, { "--diag-suppress=177" }) // disable unused variable warning
		.kernel("randbed")
		.instantiate()
		.configure(blocks, tpb);
	return launcher;
}

void BoundaryConditionKernel::launch(void* kernelLauncher, const std::vector<JitParam>& args)
{
	// prepare arguments for kernel launch
	int domainPitch = pitch / 4;
	std::vector<void*> params{ &d_h1, &d_qx1, &d_qy1, &d_sohle, &W, &H, &domainPitch };
	for (auto it = args.begin(); it != args.end(); it++)
		params.push_back((void*)&*it);
	// launch the kernel
	auto launcher = static_cast<jitify::KernelLauncher*>(kernelLauncher);
	launcher->launch(params);
}
