#include <string>
#include <sstream>
#include <fstream>

namespace utils
{
	static void replaceAll(std::string& str, const std::string& from, const std::string& to)
	{
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos)
		{
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}

	static std::string miniFormat(std::string str, const std::initializer_list<std::string>& args)
	{
		for (int i = 0; i < args.size(); i++)
		{
			const std::string& arg = *(args.begin() + i);
			replaceAll(str, "{" + std::to_string(i) + "}", arg);
		}
		return str;
	}

	static double solveSimpleMathExpression(const std::string& expression)
	{
		if (expression.find_first_not_of("0123456789.* ") != std::string::npos)
			throw std::invalid_argument("Unsupported math expression " + expression);
		
		double result = 1.0;
		std::stringstream ss(expression);
		std::string item;
		
		while (std::getline(ss, item, '*')) 
		{
			result *= std::stof(item);
		}
		return result;
	}

	static void loadFileIntoArray(const std::string& fileName, void* arr, int size, int offset)
	{
		std::ifstream file;
		file.open(fileName, std::ios::binary);
		if (!file)
			throw std::invalid_argument("Failed to open file " + fileName);

		file.seekg(offset);
		file.read(static_cast<char*>(arr), size);
		file.close();
	}
}