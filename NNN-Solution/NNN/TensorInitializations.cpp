#include "pch.h"
#include "Tensor.h"

std::vector<int> Tensor::compute_strides(const std::vector<int>& dims)
{
	int n = (int)dims.size();
	std::vector<int> strides(n);

	strides[n - 1] = 1;
	for (int i = n - 2; i >= 0; i--)
	{
		strides[i] = strides[i + 1] * dims[i + 1];
	}

	return strides;
}

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<int>& dims, bool req_grad) : _dimensions(dims), requires_grad(req_grad)
{
	_strides = compute_strides(dims);

	int size = 1;
	for (int dim : dims)
	{
		size *= dim;
	}

	_data.assign(size, 0.0);
	
	if (requires_grad)
	{
		_grad.assign(size, 0.0);
	}
}