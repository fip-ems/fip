#include "timeseries.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

TimeSeries::TimeSeries(const std::map<float, float>& series)
{
	this->series = series;
}

float TimeSeries::interp(float time)
{
	auto upper = series.lower_bound(time);
	if (upper == series.begin())
		return upper->second;
	if (upper == series.end())
		return series.rbegin()->second;

	auto lower = std::prev(upper);
	float dt = upper->first - lower->first;
	float f = (time - lower->first) / dt;
	return f * upper->second + (1.f - f) * lower->second;
}

TimeSeries* TimeSeries::fromCsv(std::string fileName, int timeColumn, int dataColumn, float timeScale, bool skipHeader)
{
	std::ifstream f(fileName);

	if (!f.is_open())
		throw std::invalid_argument("Could not open file " + fileName);

	auto isFloat = [](std::string str) {
		std::istringstream iss(str);
		float f;
		iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
		// Check the entire string was consumed and if either failbit or badbit is set
		return iss.eof() && !iss.fail();
	};

	std::map<float, float> series;
	std::string line;
	time_t firstTimestamp = -1;
	if (skipHeader)
		std::getline(f, line);
	while (std::getline(f, line))
	{
		std::stringstream ss(line);
		std::string val;
		float time, data;
		int counter = 0;
		while (std::getline(ss, val, ';'))
		{
			// parse time value
			if (counter == timeColumn)
			{
				// seconds as float
				if (isFloat(val))
					time = std::stof(val) * timeScale;
				// datetime format 
				else
				{
					std::tm t = { 0 };
					std::istringstream ss(val);
					ss >> std::get_time(&t, "%d.%m.%Y %H:%M:%S");
					if (ss.fail())
						throw std::invalid_argument("Wrong date format in timeseries " + fileName + ". Expected %d.%m.%Y %H:%M:%S");
					auto timestamp = std::mktime(&t);
					if (firstTimestamp == -1) 
						firstTimestamp = timestamp;
					time = static_cast<float>(timestamp - firstTimestamp);
				}
			}
			// parse data value
			if (counter == dataColumn)
				data = std::stof(val);

			counter++;
		}
		series.emplace(time, data);
	}
	return new TimeSeries(series);
}