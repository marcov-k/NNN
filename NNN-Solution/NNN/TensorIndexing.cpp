#include "pch.h"
#include "Tensor.h"

double& Tensor::operator[](int index)
{
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

	return _data[index];
}

const double& Tensor::operator[](int index) const
{
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

	return _data[index];
}

double& Tensor::at(const std::vector<int>& indices)
{
	int linear = linear_index(indices);
	if (linear < 0 || linear >= element_count())
	{
		throw std::out_of_range("Indices out of bounds.");
	}

	return _data[linear];
}

const double& Tensor::at(const std::vector<int>& indices) const
{
	int linear = linear_index(indices);
	if (linear < 0 || linear >= element_count())
	{
		throw std::out_of_range("Indices out of bounds.");
	}

	return _data[linear];
}

int Tensor::linear_index(const std::vector<int>& indices) const
{
	int offset = 0;

	for (int i = 0; i < (int)indices.size(); i++)
	{
		offset += indices[i] * _strides[i];
	}

	return offset;
}

const std::vector<int> Tensor::get_full_indices(int index) const
{
	int dimCount = rank();

	std::vector<int> indices(dimCount);
	for (int i = dimCount - 1; i >= 0; i--)
	{
		indices[i] = index % _dimensions[i];
		index /= _dimensions[i];
	}
	
	return indices;
}

void Tensor::get_full_indices(int index, int* __restrict indices) const
{
	int dimCount = rank();

	for (int i = dimCount - 1; i >= 0; i--)
	{
		indices[i] = index % _dimensions[i];
		index /= _dimensions[i];
	}
}