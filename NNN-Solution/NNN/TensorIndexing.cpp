#include "pch.h"
#include "Tensor.h"

/* Indexing/data access */

double& Tensor::operator[](int index)
{
	// Ensure valid index
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

	return _data[index];
}

const double& Tensor::operator[](int index) const
{
	// Ensure valid index
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

	return _data[index];
}

double& Tensor::at(const std::vector<int>& indices)
{
	return _data[linear_index(indices)];
}

const double& Tensor::at(const std::vector<int>& indices) const
{
	return _data[linear_index(indices)];
}

int Tensor::linear_index(const std::vector<int>& indices) const
{
	int offset = 0;

	for (int i = 0; i < (int)indices.size(); i++)
	{
		offset += indices[i] * _strides[i];
	}

	// Ensure valid index
	if (offset < 0 || offset >= element_count())
	{
		throw std::out_of_range("Indices out of bounds.");
	}

	return offset;
}

const std::vector<int> Tensor::get_full_indices(int index) const
{
	// Ensure valid index
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

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
	// Ensure valid index
	if (index < 0 || index >= element_count())
	{
		throw std::out_of_range("Index out of bounds.");
	}

	int dimCount = rank();

	for (int i = dimCount - 1; i >= 0; i--)
	{
		indices[i] = index % _dimensions[i];
		index /= _dimensions[i];
	}
}