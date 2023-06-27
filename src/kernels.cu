#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "kernels.h"
#include "helper_cuda.h"
#include "simdata.h"

#define WARPSZ 32
#define UNROLL_TIMELOOP 0
#define ENABLE_MASK 1

// Simulation data
namespace SimData
{
	const size_t num_warps_per_tb = 4;
	const size_t num_block_rows = 50;
	const size_t num_block_cols = WARPSZ - 4 - UNROLL_TIMELOOP * 2;
}

constexpr float DH = 0.01f;
constexpr float EPS_TROCKEN = 10e-7f;
constexpr float CN_DH_VEL = 0.01f;
constexpr float CN_DH_FRI = 0.01f;
#define G SimData::G


__forceinline__ __device__ float fp32div_wrapper(float dividend, float divisor)
{
	return (dividend == 0.0f) ? 0.0f : (dividend / divisor);
}

__forceinline__ __device__ float compute_rmg_gx(float h_i, float h_north, float qx_i, float qx_north, float qy_i)
{
	float ah_qy = max(max(DH, h_i), h_north);
	float uy_i = fp32div_wrapper(qy_i, ah_qy);
	float uy_east = __shfl_down_sync(0xffffffff, uy_i, 1);
	float uy_h = (uy_i + uy_east) * .5f;
	if (uy_i + uy_east > 0) return qx_i * uy_h;
	else return qx_north * uy_h;
}

__forceinline__ __device__ float compute_rmg_fy(float h_i, float h_east, float h_north, float h_northeast, float qx_i, float qx_north, float qy_i, float qy_east)
{
	float ah_qx = max(max(DH, h_i), h_east);
	float ah_qx_north = max(max(DH, h_north), h_northeast);
	float ux_i = fp32div_wrapper(qx_i, ah_qx);
	float ux_north = fp32div_wrapper(qx_north, ah_qx_north);
	float ux_h = (ux_i + ux_north) * .5f;
	if (ux_i + ux_north > 0) return qy_i * ux_h;
	else return qy_east * ux_h;
}

__forceinline__ __device__ float compute_rmg_fx(float h_i, float h_east, float qx_west, float qx_i)
{
	float ah_qx = max(max(DH, h_i), h_east);
	float ux_i = fp32div_wrapper(qx_i, ah_qx);
	float ux_west = __shfl_up_sync(0xffffffff, ux_i, 1);
	if (ux_west + ux_i > 0) return qx_west * ux_west;
	else return qx_i * ux_i;
}

__forceinline__ __device__ float compute_rmg_gy(float h_i, float h_north, float h_south, float qy_i, float qy_south)
{
	float ah_qy = max(max(DH, h_i), h_north);
	float ah_qy_south = max(max(DH, h_south), h_i);
	float uy_i = fp32div_wrapper(qy_i, ah_qy);
	float uy_south = fp32div_wrapper(qy_south, ah_qy_south);
	if (uy_i + uy_south > 0) return qy_south * uy_south;
	else return qy_i * uy_i;
}

__forceinline__ __device__ float compute_rmg_qx(float fx_east, float fx_i, float gx_i, float gx_south,
	float dt, float dx, float dt_inverse, float dx_inverse, float kSt,
	float h_i, float h_east, float sohle_east, float sohle_i,
	float qx_old_i, float qy_old_i, float qy_old_east, float qy_old_south, float qy_old_southeast)
{
	float dfx = (fx_east - fx_i) * dx_inverse;
	float dgx = (gx_i - gx_south) * dx_inverse;

	float ah_qx = (h_i + h_east) * .5f;
	ah_qx = max(0.f, ah_qx);
	float dhg = ah_qx * G * (-sohle_east + sohle_i) * dx_inverse;

	float dhx = (0.5f * G * h_east * h_east - 0.5f * G * h_i * h_i) * dx_inverse;

	ah_qx = max(ah_qx, DH);
	const float ah_qx_inverted = 1.f / ah_qx;
	const float ux = qx_old_i;// *ah_qx_inverted;
	const float uy = (qy_old_i + qy_old_east + qy_old_south + qy_old_southeast) * 0.25f;// *ah_qx_inverted;
	float reib = ((ux == 0.f) && (uy == 0.f)) ? 0.f :
		(dt * G / (kSt * kSt) * sqrtf(ux * ux + uy * uy) * powf(ah_qx_inverted, 2.33f));

	float qx_new = fp32div_wrapper(qx_old_i + dt * (-dfx - dgx - dhx + dhg), (1.f + reib));
	qx_new = min(qx_new, h_i * dt_inverse * dx * 0.25f);
	qx_new = max(qx_new, -h_east * dt_inverse * dx * 0.25f);
	return qx_new;
}

__forceinline__ __device__ float compute_rmg_qy(float fy_i, float fy_west, float gy_i, float gy_north,
	float dt, float dx, float dt_inverse, float dx_inverse, float kSt,
	float h_i, float h_north, float sohle_north, float sohle_i,
	float qy_old_i, float qx_old_i, float qx_old_west, float qx_old_north, float qx_old_northwest)
{
	float dgy = (gy_north - gy_i) * dx_inverse;
	float dfy = (fy_i - fy_west) * dx_inverse;

	float ah_qy = (h_i + h_north) * .5f;
	ah_qy = max(0.f, ah_qy);
	float dhg = ah_qy * G * (sohle_i - sohle_north) * dx_inverse;

	float dhy = (0.5f * G * h_north * h_north - 0.5f * G * h_i * h_i) * dx_inverse;

	ah_qy = max(ah_qy, DH);
	float ah_qy_inverted = 1.f / ah_qy;
	float ux = (qx_old_i + qx_old_north + qx_old_west + qx_old_northwest) * 0.25f;// *ah_qy_inverted;
	float uy = qy_old_i;// *ah_qy_inverted;
	float reib = ((ux == 0.f) && (uy == 0.f)) ? 0.f :
		(dt * G / (kSt * kSt) * sqrtf(ux * ux + uy * uy) * powf(ah_qy_inverted, 2.33f));

	float qy_new = fp32div_wrapper(qy_old_i + dt * (-dgy - dfy - dhy + dhg), (1 + reib));
	qy_new = min(qy_new, h_i * dt_inverse * dx * 0.25f);
	qy_new = max(qy_new, -h_north * dt_inverse * dx * 0.25f);
	return qy_new;
}

__forceinline__ __device__ uint64_t eval_mask(uint64_t* mask, int stencil, 
	int row, int col, int W, int H, int rowsPerBlock, 
	int laneId, int global_warpIdx, int warpsPerRow, 
	int& firstRowToCompute, int& lastRowToCompute)
{
	uint64_t m = 0;
	if (laneId == 0)
	{
		int maskIdx = blockIdx.y * warpsPerRow + global_warpIdx;
		m = mask[maskIdx];
		uint64_t m_west = (col == 0) ? 0 : mask[maskIdx - 1];
		uint64_t m_east = (col == W - 1) ? 0 : mask[maskIdx + 1];
		uint64_t m_north = (m >> 1);
		uint64_t m_south = (m << 1);
		m_north |= (row == H - 1) ? 0 : ((mask[maskIdx + warpsPerRow] & (1 << stencil)) << (rowsPerBlock - 1));
		m_south |= (row == 0) ? 0 : (mask[maskIdx - warpsPerRow] >> (rowsPerBlock - 1));
		m = m | m_south | m_north | m_west | m_east;
	}
	m = __shfl_sync(0xffffffff, m, 0);
	if (m != 0)
	{
		firstRowToCompute = max(0, __ffsll(m) - stencil - 1);
		lastRowToCompute = min(rowsPerBlock + stencil, 64 - __clzll(m));
	}
	return m;
}

__forceinline__ __device__ void store_mask(
	uint64_t* mask, uint64_t mask_new, int tx, int wx, int totalrows, int firstRowToCompute, int warpsPerRow)
{
#if ENABLE_MASK
	// shift mask such that first row in compute block is first bit in mask
	if (totalrows < sizeof(uint64_t) * 8)
		mask_new >>= sizeof(uint64_t) * 8 - (totalrows + firstRowToCompute);

	// combine 32 columns of mask_new to one mask_new valid for warp's whole compute block
	for (int offset = 16; offset > 0; offset /= 2)
		mask_new |= __shfl_down_sync(0xffffffff, mask_new, offset);

	if (tx == 0)
		mask[blockIdx.y * warpsPerRow + wx] = mask_new;
#endif
}

__forceinline__ __device__ void store_courant(float* timesteps, float dt_cfl_min, int stencil, int tx, int wx, int W, int warpsPerRow)
{
	const auto num_active_threads = wx * (32 - stencil * 2);
	const auto boundary_distance = W - num_active_threads;
	if (boundary_distance < 32)
	{
		for (int i = 1; i < boundary_distance; i++)
			dt_cfl_min = min(dt_cfl_min, __shfl_sync(0xffffffff, dt_cfl_min, i));
		if (tx == 0)
			timesteps[blockIdx.y * warpsPerRow + wx] = dt_cfl_min;
	}
	else
	{
		// find minimum courant number across all 32 threads in warp
		for (int offset = 16; offset > 0; offset /= 2)
			dt_cfl_min = min(dt_cfl_min, __shfl_down_sync(0xffffffff, dt_cfl_min, offset));
		if (tx == 0)
			timesteps[blockIdx.y * warpsPerRow + wx] = dt_cfl_min;
	}
}

__forceinline__ __device__ bool is_negative_zero(float x)
{
	return (*(int*)&x) == 0x80000000;
}

__global__ void timestep_reduce(float* timesteps, float* cfl_ts, int N)
{
	float min_ts = 1.f;
	int argmin = threadIdx.x;

	__shared__ float min_ts_per_warp[32];
	__shared__ float argmin_per_warp[32];

	for (int i = 0; i < N; i += blockDim.x)
	{
		float ts = i + threadIdx.x < N ? timesteps[i + threadIdx.x] : 1.f;
		int tid = i + threadIdx.x;

		for (int offset = 16; offset > 0; offset /= 2)
		{
			float neighbor_ts = __shfl_down_sync(0xffffffff, ts, offset);
			int neighbor_tid = __shfl_down_sync(0xffffffff, tid, offset);
			tid = ts < neighbor_ts ? tid : neighbor_tid;
			ts = ts < neighbor_ts ? ts : neighbor_ts;
		}

		if (threadIdx.x % 32 == 0)
		{
			min_ts_per_warp[threadIdx.x / 32] = ts;
			argmin_per_warp[threadIdx.x / 32] = tid;
		}

		__syncthreads();

		if (threadIdx.x < 32)
		{
			float ts = min_ts_per_warp[threadIdx.x];
			int tid = argmin_per_warp[threadIdx.x];
			for (int offset = 16; offset > 0; offset /= 2)
			{
				float neighbor_ts = __shfl_down_sync(0xffffffff, ts, offset);
				int neighbor_tid = __shfl_down_sync(0xffffffff, tid, offset);
				tid = ts < neighbor_ts ? tid : neighbor_tid;
				ts = ts < neighbor_ts ? ts : neighbor_ts;
			}
			if (threadIdx.x == 0 && ts < min_ts)
			{
				min_ts = ts;
				argmin = tid;
			}
		}
	}

	if (threadIdx.x == 0)
	{
		*cfl_ts = min_ts;
		*(cfl_ts + 1) = argmin;
	}
}

__global__ void flood_plains_kernel(float* h, float* h_max, unsigned W, unsigned H, unsigned pitch)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= W || row >= H) 
		return;
	const int idx = row * pitch + col;
	h_max[idx] = max(h_max[idx], h[idx]);
}

template<unsigned BLOCKROWS, bool PRECIP>
__global__ void rmg_kernel(
	float* h1, float* h2,
	float* qx1, float* qx2, float* qy1, float* qy2,
	float* sohle, 
	float* rei, float kSt,
	uint64_t* mask,
	float dx, float dt,
	unsigned W, unsigned H, unsigned pitch,
	float* timesteps,
	float no_value)
{
	const float dx_inverse = 1.f / dx;
	const float dt_inverse = 1.f / dt;
	const int warpsPerBlock = blockDim.x / 32;
	const int warpsPerRow = gridDim.x * warpsPerBlock;
	const int tx = threadIdx.x % 32;
	const int wx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);
	const int stencil = 2;
	const int col = wx * (32 - stencil * 2) + tx;
	int row = blockIdx.y * BLOCKROWS;

	// cfl
	float dt_cfl_min = 1.f;

	if (col >= W || row >= H)
		return;

#if ENABLE_MASK
	uint64_t mask_new = 0;
	int firstRowToCompute = 0;
	int lastRowToCompute = BLOCKROWS + stencil;
	if constexpr (!PRECIP) // disable mask when precip is applied
	{
		if (!eval_mask(mask, stencil, row, col, W, H, BLOCKROWS, tx, wx, warpsPerRow, firstRowToCompute, lastRowToCompute))
			return;
	}
#else
	constexpr int firstRowToCompute = 0;
	constexpr int lastRowToCompute = BLOCKROWS + 2;
#endif

	row += firstRowToCompute;
	int global_idx = row * pitch + col;
	const int totalrows = min(lastRowToCompute - firstRowToCompute, H - row);


	float h_i = h1[global_idx];
	float qx1_i = qx1[global_idx];
	float qy1_i = qy1[global_idx];
	float sohle_north = sohle[global_idx + pitch];
	float h_north = h1[global_idx + pitch];
	float h_northnorth = h1[global_idx + 2 * pitch];
	float qy1_north = qy1[global_idx + pitch];
	float qx1_north = qx1[global_idx + pitch];
	float h_east = __shfl_down_sync(0xffffffff, h_i, 1);
	float h_west = __shfl_up_sync(0xffffffff, h_i, 1);
	float h_northeast = __shfl_down_sync(0xffffffff, h_north, 1);
	float qx1_northwest = __shfl_up_sync(0xffffffff, qx1_north, 1);
	float qy1_east = __shfl_down_sync(0xffffffff, qy1_i, 1);
	// not needed in first iteration
	float rb = 0;
	float sohle_i = 0;
	float sohle_east = 0;
	float qy1_south = 0;
	float qx1_west = 0;
	float qy1_southeast = 0;
	float gx_south = 0.f;
	float gy_i = 0.f;
	float qy2_south = 0.f;
	float uy_south = 0.f;

	for (int i = 0; i < totalrows; row++, global_idx += pitch, i++)
	{
#if ENABLE_MASK
		mask_new >>= 1;
#endif
		float rb_north = 0.f, sohle_northnorth = 0.f, ah_northnorthnorth = 0.f, qx1_northnorth = 0.f, qy1_northnorth = 0.f;
		if (i != totalrows - 1)
		{
			rb_north = rei ? rei[global_idx + pitch] : kSt;
			sohle_northnorth = sohle[global_idx + 2 * pitch];
			ah_northnorthnorth = h1[global_idx + 3 * pitch];
			qx1_northnorth = qx1[global_idx + 2 * pitch];
			qy1_northnorth = qy1[global_idx + 2 * pitch];
		}

		float ah_qx = max(max(DH, h_i), h_east);
		float ah_qy = max(max(DH, h_i), h_north);
		float ux_i = fp32div_wrapper(qx1_i, ah_qx);
		float uy_i = fp32div_wrapper(qy1_i, ah_qy);
		float ux_west = __shfl_up_sync(0xffffffff, ux_i, 1);
		float fx_i = compute_rmg_fx(h_i, h_east, qx1_west, qx1_i);
		float fy_i = compute_rmg_fy(h_i, h_east, h_north, h_northeast, qx1_i, qx1_north, qy1_i, qy1_east);
		float gx_i = compute_rmg_gx(h_i, h_north, qx1_i, qx1_north, qy1_i);
		float gy_north = compute_rmg_gy(h_north, h_northnorth, h_i, qy1_north, qy1_i);
		float fx_east = __shfl_down_sync(0xffffffff, fx_i, 1);
		float fy_west = __shfl_up_sync(0xffffffff, fy_i, 1);

		float qx2_i = compute_rmg_qx(fx_east, fx_i, gx_i, gx_south,
			dt, dx, dt_inverse, dx_inverse, rb,
			h_i, h_east, sohle_east, sohle_i,
			qx1_i, qy1_i, qy1_east, qy1_south, qy1_southeast);
		float qy2_i = compute_rmg_qy(fy_i, fy_west, gy_i, gy_north,
			dt, dx, dt_inverse, dx_inverse, rb,
			h_i, h_north, sohle_north, sohle_i,
			qy1_i, qx1_i, qx1_west, qx1_north, qx1_northwest);

		// for flow boundary conditions
		if (col == 0 || col == W - 2)
			qx2_i = qx1_i;
		if (row == 0 || row == H - 2)
			qy2_i = qy1_i;

		float qx2_west = __shfl_up_sync(0xffffffff, qx2_i, 1);

		//////////////////// WATER LEVEL
		float h2_i = h_i - dt * dx_inverse * ((qx2_i - qx2_west) + (qy2_i - qy2_south));

		if (sohle_i != no_value)
		{
			// only the values of inner threads are computed correctly, due to numerical scheme's stencil
			const bool innerRow = i >= stencil && row < H - stencil;
			const bool innerCol = tx >= stencil && tx < 32 - stencil && col < W - stencil;
			if ((innerCol || col == 1 || col == W - 2) && (innerRow || row == 1 || row == H - 2))
			{
				h2[global_idx] = h2_i;

				if (h2_i > EPS_TROCKEN)
				{
					float umax = max(fabsf(uy_i), max(fabsf(uy_south), max(fabsf(ux_i), fabsf(ux_west))));
					float dt_cfl = dx / (sqrtf(G * h2_i) + umax);
					dt_cfl_min = min(dt_cfl_min, dt_cfl);
#if ENABLE_MASK
					mask_new |= 0x8000000000000000;
#endif
				}
			}
			if ((innerCol || col == 1) && innerRow)
				qx2[global_idx] = qx2_i;
			if ((innerRow || row == 1) && innerCol)
				qy2[global_idx] = qy2_i;
		}

		// travel north
		{
			qy2_south = qy2_i;
			qy1_south = qy1_i;
			gx_south = gx_i;
			qy1_southeast = qy1_east;
			uy_south = uy_i;
			h_i = h_north;
			sohle_i = sohle_north;
			rb = rb_north;
			qx1_i = qx1_north;
			qy1_i = qy1_north;
			gy_i = gy_north;
			h_east = h_northeast;
			qx1_west = qx1_northwest;
			h_north = h_northnorth;
			qx1_north = qx1_northnorth;
			qy1_north = qy1_northnorth;
			sohle_north = sohle_northnorth;
			h_northnorth = ah_northnorthnorth;
			h_west = __shfl_up_sync(0xffffffff, h_i, 1);
			qy1_east = __shfl_down_sync(0xffffffff, qy1_i, 1);
			sohle_east = __shfl_down_sync(0xffffffff, sohle_i, 1);
			h_northeast = __shfl_down_sync(0xffffffff, h_north, 1);
			qx1_northwest = __shfl_up_sync(0xffffffff, qx1_north, 1);
		}
	}

	store_courant(timesteps, dt_cfl_min, 2, tx, wx, W, warpsPerRow);
	store_mask(mask, mask_new, tx, wx, totalrows, firstRowToCompute, warpsPerRow);
}


void launchComputeKernel()
{
	using namespace SimData;

	dim3 block(num_warps_per_tb * WARPSZ, 1);
	dim3 grid(W / (num_block_cols * num_warps_per_tb) + 1, H / num_block_rows + 1);

	rmg_kernel<num_block_rows, false> <<< grid, block >>>
		(d_h1, d_h2, d_qx1, d_qx2, d_qy1, d_qy2, d_sohle,
			d_rei, kSt_fixed, d_mask, dx, dt, W, H, pitch / 4,
			d_timesteps,
			invalid_terrain);

	duration += dt;
}

void launchTimestepReduceKernel()
{
	using namespace SimData;
	timestep_reduce <<< 1, 1024 >>> (d_timesteps, h_cfl_ts, num_compute_blocks);
}

void launchFloodPlainKernel()
{
	using namespace SimData;
	dim3 block(32, 32);
	dim3 grid(W / 32 + 1, H / 32 + 1);
	flood_plains_kernel <<< grid, block >>> (d_h1, d_hmax, W, H, pitch / 4);
}