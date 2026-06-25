#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

class Tensor
{
public:
	const std::vector<double>& data() const
	{
		return _data;
	}
	const std::vector<double>& grad() const
	{
		return _grad;
	}
	const std::vector<int>& dimensions() const
	{
		return _dimensions;
	}
	const std::vector<int>& strides() const
	{
		return _strides;
	}
	int rank() const
	{
		return (int)_dimensions.size();
	}
	int element_count() const
	{
		return (int)_data.size();
	}
	int grad_count() const
	{
		return (int)_grad.size();
	}
	bool requires_grad = false;
	static bool inference;

	Tensor();

	Tensor(const std::vector<int>& dims, bool req_grad = false);

	double& operator[](int index);

	const double& operator[](int index) const;

	double& at(const std::vector<int>& indices);

	const double& at(const std::vector<int>& indices) const;

	int linear_index(const std::vector<int>& indices) const;

	int linear_index(std::span<int> indices) const;

	const std::vector<int> get_full_indices(int index) const;

private:
	std::vector<double> _data;
	std::vector<double> _grad;
	std::vector<int> _dimensions;
	std::vector<int> _strides;
	static constexpr int VECTOR_SIZE = 256 / 64; // AVX2 register width / bits per double
	static constexpr int PARALLEL_THRESHOLD = 500000;
	std::vector<std::shared_ptr<Tensor>> _parents;
	std::vector<std::weak_ptr<Tensor>> _results;
	int _op_index = 0;
	std::function<void()> _backward = [] {};
	static int _forward_gen;
	int _last_gen = -1;
	std::optional<std::vector<std::shared_ptr<Tensor>>> _topo;
	std::optional<std::unordered_set<std::shared_ptr<Tensor>>> _visited;

	static std::vector<int> compute_strides(const std::vector<int>& dims);

	void get_full_indices(int index, std::span<int> indices) const;
};