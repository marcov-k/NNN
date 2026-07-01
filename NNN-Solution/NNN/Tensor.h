#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <immintrin.h>
#include <limits>
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
	std::vector<double>& mutable_data()
	{
		return _data;
	}
	const std::vector<double>& grad() const
	{
		return _grad;
	}
	std::vector<double>& mutable_grad()
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
	static bool log_debug;

	Tensor();

	Tensor(const std::vector<int> dims, bool req_grad = false);

	Tensor(double value, const std::vector<int> dims, bool req_grad = false);

	static std::shared_ptr<Tensor> init_weights(int input_count, int neuron_count);

	static std::shared_ptr<Tensor> init_biases(int neuron_count);

	static std::shared_ptr<Tensor> init_kernels(int filter_count, const std::vector<int>& kernel_dims, int input_channels);

	std::shared_ptr<Tensor> copy() const;

	double& operator[](int index);

	const double& operator[](int index) const;

	double& at(const std::vector<int>& indices);

	const double& at(const std::vector<int>& indices) const;

	int linear_index(const std::vector<int>& indices) const;

	const std::vector<int> get_full_indices(int index) const;

	void restore_grad();

	void clear_graph();

	static void begin_forward();

	void backward();

	static std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, double b);

	static std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, double b);

	static std::shared_ptr<Tensor> sub(double a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, double b);

	static std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, double b);

	static std::shared_ptr<Tensor> div(double a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& exp);

	static std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, double exp);

	static std::shared_ptr<Tensor> pow(double a, const std::shared_ptr<Tensor>& exp);

	static std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& arg, const std::shared_ptr<Tensor>& log_base);

	static std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& arg, double log_base);

	static std::shared_ptr<Tensor> log(double arg, const std::shared_ptr<Tensor>& log_base);

	static std::shared_ptr<Tensor> ln(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	static std::shared_ptr<Tensor> convolve(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& kernels,
		const std::shared_ptr<Tensor>& biases);

	static std::shared_ptr<Tensor> mask_actions(const std::shared_ptr<Tensor>& q_values, const std::vector<int>& actions);

	static int arg_max(const std::shared_ptr<const Tensor>& t);

	static std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& t, const std::vector<int>& axes);

	static std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& t, const std::vector<int>& target_dims);

	static std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& t, const std::vector<int>& new_dims);

	static std::shared_ptr<Tensor> flatten(const std::shared_ptr<Tensor>& t, int start_axis = 0);

	static std::shared_ptr<Tensor> wrap_batch(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> clip(const std::shared_ptr<Tensor>& t, double min, double max);

	static std::shared_ptr<Tensor> get_dense_dropout_mask(const std::vector<int>& dims, double dropout);

	static std::shared_ptr<Tensor> get_spatial_dropout_mask(const std::vector<int>& dims, double dropout);

	static std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> leaky_relu(const std::shared_ptr<Tensor>& t, double tau);

	static std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t);

	static std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target);

	static std::shared_ptr<Tensor> huber(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target, double delta);

	static std::shared_ptr<Tensor> softmax_cross_entropy(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target);

private:
	std::vector<double> _data;
	std::vector<double> _grad;
	std::vector<int> _dimensions;
	std::vector<int> _strides;
	static constexpr int VECTOR_SIZE = 256 / 64; // AVX2 register width / bits per double
	static constexpr long PARALLEL_THRESHOLD = 50000;
	std::vector<std::shared_ptr<Tensor>> _parents;
	std::vector<std::shared_ptr<Tensor>> _results;
	int _op_index = 0;
	std::function<void()> _backward = [] {};
	static int _forward_gen;
	int _last_gen = -1;
	std::optional<std::vector<std::shared_ptr<Tensor>>> _topo = {};
	std::optional<std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>> _visited = {};

	static std::vector<int> compute_strides(const std::vector<int>& dims);

	void get_full_indices(int index, int* __restrict indices) const;

	void prepare_forward();

	void finalize_forward();

	static void build_topo(const std::shared_ptr<Tensor>& t, std::vector<std::shared_ptr<Tensor>>& topo,
		std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>& visited);

	static std::shared_ptr<Tensor> get_result_tensor(const std::shared_ptr<Tensor>& owner, const std::vector<int>& dims, bool requires_grad);

	static void transpose_matrix(const double* __restrict src, double* __restrict dst, int src_off, int dst_off,
		int rows, int cols);

	static void compute_row(int i, int n, int p, const double* __restrict a, const double* __restrict b_t,
		double* __restrict r, int a_off, int b_t_off, int r_off);

	static void compute_output_position(int batch_out_pos, int spatial_rank, int out_spatial_size, int filter_count,
		int kernel_spatial_size, int input_channels, const int* __restrict out_spatial_strides,
		const int* __restrict kernel_spatial_strides, const int* __restrict input_strides,
		const int* __restrict kernel_strides, const int* __restrict result_strides, const double* __restrict input_data,
		const double* __restrict kernel_data, const double* __restrict bias_data, double* __restrict result_data);

	static void compute_kernel_grad(int fkp, int spatial_rank, int batches, int out_spatial_size,
		int kernel_spatial_size, int input_channels, const int* __restrict out_spatial_strides,
		const int* __restrict kernel_spatial_strides, const int* __restrict input_strides,
		const int* __restrict kernel_strides, const int* __restrict result_strides, const double* __restrict input_data,
		double* __restrict kernel_grad, const double* __restrict result_grad);

	static void compute_input_grad(int batch_in_pos, int spatial_rank, int in_spatial_size, int filter_count,
		int kernel_spatial_size, int input_channels, const int* __restrict in_spatial_strides,
		const int* __restrict kernel_spatial_strides, const int* __restrict out_spatial_dims,
		const int* __restrict input_strides, const int* __restrict kernel_strides, const int* __restrict result_strides,
		double* __restrict input_grad, const double* __restrict kernel_data, const double* __restrict result_grad);
};

// Free function operators

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::add(a, b);
}

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::add(a, b);
}

inline std::shared_ptr<Tensor> operator+(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::add(b, a);
}

inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::sub(a, b);
}

inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::sub(a, b);
}

inline std::shared_ptr<Tensor> operator-(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::sub(a, b);
}

inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::mul(a, b);
}

inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::mul(a, b);
}

inline std::shared_ptr<Tensor> operator*(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::mul(b, a);
}

inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::div(a, b);
}

inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::div(a, b);
}

inline std::shared_ptr<Tensor> operator/(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::div(a, b);
}