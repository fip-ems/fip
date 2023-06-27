#include "boundary_condition.h"
#include "simdata.h"
#include <math.h>
#include <stdexcept>

float BoundaryCondition::sumWaterDepths(float z_soll)
{
	float ah_sum = 0.f;
	int idx_inc = getIdxForNthBoundaryCell(1) - getIdxForNthBoundaryCell(0);
	for (int idx = getIdxForNthBoundaryCell(from); idx < getIdxForNthBoundaryCell(to); idx += idx_inc)
	{
		float ah = z_soll - SimData::h_sohle[idx];
		if (ah < 0.01f) ah = 0.f;
		ah_sum += sqrtf(SimData::G * ah) * ah;
	}
	ah_sum *= SimData::dx;
	return ah_sum;
}

BoundaryCondition::BoundaryCondition(RandType type, RandSide side, int from, int to, float z, float q)
	: type(type), side(side), from(from), to(to), z(z), q(q)
{
	if (side == RandSide::left)
	{
		getIdxForNthBoundaryCell = [](int n) { return n * SimData::W; };
		getIdxForNthBoundaryCellNeighbor = [&](int n) { return getIdxForNthBoundaryCell(n) + 1; };
	}
	if (side == RandSide::right)
	{
		getIdxForNthBoundaryCell = [](int n) { return n * SimData::W + (SimData::W - 1); };
		getIdxForNthBoundaryCellNeighbor = [&](int n) { return getIdxForNthBoundaryCell(n) - 1; };
	}
	if (side == RandSide::top)
	{
		getIdxForNthBoundaryCell = [](int n) { return n + (SimData::H - 1) * SimData::W; };
		getIdxForNthBoundaryCellNeighbor = [&](int n) { return getIdxForNthBoundaryCell(n) - SimData::W; };
	}
	if (side == RandSide::bottom)
	{
		getIdxForNthBoundaryCell = [](int n) { return n; };
		getIdxForNthBoundaryCellNeighbor = [&](int n) { return getIdxForNthBoundaryCell(n) + SimData::W; };
	}
}

float BoundaryCondition::getZ(double duration)
{
	return z;
}

float BoundaryCondition::getQ(double duration)
{
	if (type != RandType::q)
		return q;
	return q / sumWaterDepths(z);
}

BoundaryConditionTimeseries::BoundaryConditionTimeseries(RandType type, RandSide side, int from, int to, TimeSeries* z_timeseries, TimeSeries* q_timeseries)
	: BoundaryCondition(type, side, from, to, 0, 0), z_timeseries(z_timeseries), q_timeseries(q_timeseries)
{
	if (type == RandType::z && !z_timeseries)
		throw std::invalid_argument("BoundaryCondition error: type z but no z timeseries given");
	if (type == RandType::q && (!z_timeseries || !q_timeseries))
		throw std::invalid_argument("BoundaryCondition error: type q requires both z and q timeseries");
}

float BoundaryConditionTimeseries::getZ(double duration)
{
	return z_timeseries->interp(duration);
}

float BoundaryConditionTimeseries::getQ(double duration)
{
	if (type != RandType::q)
		return 0;
	float z = getZ(duration);
	return q_timeseries->interp(duration) / sumWaterDepths(z);
}
