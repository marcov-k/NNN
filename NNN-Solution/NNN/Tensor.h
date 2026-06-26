#pragma once

#include <functional>
#include <immintrin.h>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

class Tensor; // forward declaration

struct TensorPtrHash
{
	size_t operator()(const std::shared_ptr<Tensor>& ptr) const
	{
		return std::hash<Tensor*>()(ptr.get());
	}
};

struct TensorPtrEqual
{
	bool operator()(const std::shared_ptr<Tensor>& a,
		const std::shared_ptr<Tensor>& b) const
	{
		return a.get() == b.get();
	}
};

class Tensor : public std::enable_shared_from_this<Tensor>
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

	void restore_grad();

	void clear_graph();

	static void begin_forward();

	void backward();

	static std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> t);

private:
	std::vector<double> _data;
	std::vector<double> _grad;
	std::vector<int> _dimensions;
	std::vector<int> _strides;
	static constexpr int VECTOR_SIZE = 256 / 64; // AVX2 register width / bits per double
	static constexpr int PARALLEL_THRESHOLD = 500000;
	std::vector<std::shared_ptr<Tensor>> _parents;
	std::vector<std::shared_ptr<Tensor>> _results;
	int _op_index = 0;
	std::function<void()> _backward = [] {};
	static int _forward_gen;
	int _last_gen = -1;
	std::optional<std::vector<std::shared_ptr<Tensor>>> _topo = {};
	std::optional<std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>> _visited = {};

	static std::vector<int> compute_strides(const std::vector<int>& dims);

	void get_full_indices(int index, std::span<int> indices) const;

	void prepare_forward();

	void finalize_forward();

	static void build_topo(std::shared_ptr<Tensor> t, std::vector<std::shared_ptr<Tensor>>& topo,
		std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>& visited);

	static std::shared_ptr<Tensor> get_result_tensor(std::shared_ptr<Tensor> owner, const std::vector<int>& dims, bool requires_grad);
};