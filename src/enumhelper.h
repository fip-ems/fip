#pragma once
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>

#define ENUM_DATA(Enum, ...) template<> const std::string EnumStrings<Enum>::data[] = { __VA_ARGS__ }

template<typename T>
struct EnumStrings
{
	static const std::string data[];
};

template<typename T>
static T stringToEnum(const std::string& str)
{
	static auto begin = std::begin(EnumStrings<T>::data);
	static auto end = std::end(EnumStrings<T>::data);
	auto find = std::find(begin, end, str);
	if (find == end)
		throw std::invalid_argument("Invalid enum value: " + str);
	return static_cast<T>(std::distance(begin, find));
}
