#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

/* Initialization */

// Computes the strides of the tensor using the given dimensions.
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

Tensor::Tensor(const std::vector<int> dims, bool req_grad) : _dimensions(dims), requires_grad(req_grad)
{
	_strides = compute_strides(dims);

	// Compute linear size and initialize data and gradient vectors

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

Tensor::Tensor(double value, const std::vector<int> dims, bool req_grad) : _dimensions(dims), requires_grad(req_grad)
{
	_strides = compute_strides(dims);

	// Compute linear size and initialize data and gradient vectors

	int size = 1;
	for (int dim : dims)
	{
		size *= dim;
	}

	_data.assign(size, value);

	if (requires_grad)
	{
		_grad.assign(size, 0.0);
	}
}

std::shared_ptr<Tensor> Tensor::init_weights(int input_count, int neuron_count)
{
	auto weights = std::make_shared<Tensor>(std::vector<int>{input_count, neuron_count}, true);

	// Initialize weight values using He initialization
	double std_dev = std::sqrt(2.0 / input_count);
	const int element_count = weights->element_count();
	for (int i = 0; i < element_count; ++i)
	{
		weights->_data[i] = MathUtils::next_gaussian(0.0, std_dev);
	}

	return weights;
}

std::shared_ptr<Tensor> Tensor::init_biases(int neuron_count)
{
	return std::make_shared<Tensor>(0.01, std::vector<int>{neuron_count}, true);
}

std::shared_ptr<Tensor> Tensor::init_kernels(int filter_count, const std::vector<int>& kernel_dims, int input_channels)
{
	// Compute kernels tensor dimensions
	std::vector<int> dims;
	dims.reserve(kernel_dims.size() + 2);
	dims.push_back(filter_count);
	int fan_in = input_channels;
	for (int dim : kernel_dims)
	{
		dims.push_back(dim);
		fan_in *= dim;
	}
	dims.push_back(input_channels);

	auto kernels = std::make_shared<Tensor>(dims, true);

	// Initialize kernel values using He initialization
	double std_dev = std::sqrt(2.0 / fan_in);
	const int element_count = kernels->element_count();
	for (int i = 0; i < element_count; ++i)
	{
		kernels->_data[i] = MathUtils::next_gaussian(0.0, std_dev);
	}

	return kernels;
}

std::shared_ptr<Tensor> Tensor::copy() const
{
	auto copy = std::make_shared<Tensor>(_dimensions, requires_grad);
	copy->_data = _data;
	return copy;
}