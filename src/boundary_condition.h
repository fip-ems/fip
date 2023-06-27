#pragma once

#include <functional>
#include "timeseries.h"

enum class RandType
{
	z, q, close, open
};

enum class RandSide
{
	left, right, top, bottom
};

class BoundaryCondition
{
private:
	float z, q;
protected:
	RandType type;
	RandSide side;
	int from, to;
public:
	BoundaryCondition(RandType type, RandSide side, int from, int to, float z, float q);
	float sumWaterDepths(float z_soll);
	RandType getType() { return type; }
	RandSide getSide() { return side; }
	int getFrom() { return from; }
	int getTo() { return to; }
	std::function<int(int)> getIdxForNthBoundaryCell;
	std::function<int(int)> getIdxForNthBoundaryCellNeighbor;
	virtual float getZ(double duration);
	virtual float getQ(double duration);
};

class BoundaryConditionTimeseries : public BoundaryCondition
{
private:
	TimeSeries* z_timeseries, * q_timeseries;

public:
	BoundaryConditionTimeseries(RandType type, RandSide side, int from, int to, TimeSeries* z_timeseries, TimeSeries* q_timeseries);
	float getZ(double duration) override;
	float getQ(double duration) override;
};
