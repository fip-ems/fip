#pragma once

#include <vector>
#include <string>
#include <map>

class TimeSeries
{
private:
	std::map<float, float> series;

public:
	TimeSeries(const std::map<float, float>& series);

	float interp(float time);

	static TimeSeries* fromCsv(std::string fileName, int timeColumn, int dataColumn, float timeScale = 1.f, bool skipHeader = true);

};