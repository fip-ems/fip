#pragma once
#include <vector>

union JitParam
{
	float f;
	int i;
	JitParam(float f) : f(f) {}
	JitParam(int i) : i(i) {}
};

namespace BoundaryConditionKernel
{
	extern void* generate();
	extern void launch(void* kernelLauncher, const std::vector<JitParam>& args);
}
