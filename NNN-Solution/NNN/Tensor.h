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

#include "DataContainers.h"

class Tensor; // forward declaration

// Tensor hash constructor (reference-based).
struct TensorPtrHash
{
	size_t operator()(const std::shared_ptr<Tensor>& ptr) const
	{
		return std::hash<Tensor*>()(ptr.get());
	}
};

// Tensor equality operator (reference-based).
struct TensorPtrEqual
{
	bool operator()(const std::shared_ptr<Tensor>& a,
		const std::shared_ptr<Tensor>& b) const
	{
		return a.get() == b.get();
	}
};

// Core tensor data structure - implements full autograd graph.
class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
	/* Initialization */

	// Creates a new empty tensor instance.
	Tensor();

	// Creates a new tensor instance with the given dimensions and requires_grad flag.
	Tensor(const std::vector<int> dims, bool req_grad = false);

	// Creates a new tensor instance with the given dimensions and requires_grad flag and fills it with the given scalar value.
	Tensor(double value, const std::vector<int> dims, bool req_grad = false);

	// Initializes a new weights tensor instance for a dense layer.
	static std::shared_ptr<Tensor> init_weights(int input_count, int neuron_count);

	// Initializes a new bias tensor instance.
	static std::shared_ptr<Tensor> init_biases(int neuron_count);

	// Initializes a new kernels tensor instance for a convolutional layer.
	static std::shared_ptr<Tensor> init_kernels(int filter_count, const std::vector<int>& kernel_dims, int input_channels);

	// Creates a copy instance of the tensor - detached from autograd graph.
	std::shared_ptr<Tensor> copy() const;

	/* Indexing/data access */

	// Returns a constant reference to the tensor's data vector.
	const std::vector<double>& data() const
	{
		return _data;
	}

	// Returns a mutable reference to the tensor's data vector.
	std::vector<double>& mutable_data()
	{
		return _data;
	}

	// Returns a constant reference to the tensor's gradient vector.
	const std::vector<double>& grad() const
	{
		return _grad;
	}

	// Returns a mutable reference to the tensor's gradient vector.
	std::vector<double>& mutable_grad()
	{
		return _grad;
	}

	// Returns a constant reference to the tensor's dimension vector.
	const std::vector<int>& dimensions() const
	{
		return _dimensions;
	}

	// Returns a constant reference to the tensor's stride vector.
	const std::vector<int>& strides() const
	{
		return _strides;
	}

	// Returns the tensor's rank.
	int rank() const
	{
		return (int)_dimensions.size();
	}

	// Returns the total number of data elements in the tensor.
	int element_count() const
	{
		return (int)_data.size();
	}

	// Returns the total number of gradient elements in the tensor.
	int grad_count() const
	{
		return (int)_grad.size();
	}

	// Whether the tensor's gradient must be calculated during the backward pass.
	bool requires_grad = false;

	// Whether the engine is operating in inference mode (disables autograd graph construction).
	static bool inference;

	// Whether the engine should log debug data.
	static bool log_debug;

	// Returns a mutable reference to the value at the given linear index in the tensor.
	double& operator[](int index);

	// Returns a constant reference to the value at the given lienar index in the tensor.
	const double& operator[](int index) const;

	// Returns a mutable reference to the value at the given indices in the tensor.
	double& at(const std::vector<int>& indices);

	// Returns a constant reference to the value at the given indices in the tensor.
	const double& at(const std::vector<int>& indices) const;

	// Converts indices into a linear index in the tensor.
	int linear_index(const std::vector<int>& indices) const;

	// Converts a linear index into indices in the tensor.
	const std::vector<int> get_full_indices(int index) const;

	/* Autograd graph */

	// Restores and resets the gradient vector of the tensor.
	void restore_grad();

	// Clears the autograd graph connections of the tensor.
	void clear_graph();

	// Begins a new forward pass of the autograd graph.
	static void begin_forward();

	// Performs a backward gradient-calculation pass of the autograd graph starting at this tensor.
	void backward();

	// Finalizes the current inference forward pass.
	void finalize_inference();

	/* Tensor operations - autograd graph connected */

	// Adds two tensors -> r = a + b
	static std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	// Adds a tensor and scalar -> r = a + b
	static std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, double b);

	// Subtracts two tensors -> r = a - b
	static std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	// Subtracts a tensor and scalar -> r = a - b
	static std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, double b);

	// Subtracts a scalar and tensor -> r = a - b
	static std::shared_ptr<Tensor> sub(double a, const std::shared_ptr<Tensor>& b);

	// Multiplies two tensors -> r = a * b
	static std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	// Multiplies a tensor and scalar -> r = a * b
	static std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, double b);

	// Divides two tensors -> r = a / b
	static std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	// Divides a tensor and scalar -> r = a / b
	static std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, double b);

	// Divides a scalar and tensor -> r = a / b
	static std::shared_ptr<Tensor> div(double a, const std::shared_ptr<Tensor>& b);

	// Exponentiates two tensors -> r = a ^ exp
	static std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& exp);

	// Exponentiates a tensor and scalar -> r = a ^ exp
	static std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, double exp);

	// Exponentiates a scalar and tensor -> r = a ^ exp
	static std::shared_ptr<Tensor> pow(double a, const std::shared_ptr<Tensor>& exp);

	// Computes the natural exponentiation of a tensor -> r = e ^ t
	static std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& t);

	// Computes the logarithm of two tensors -> r = log_base(arg)
	static std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& arg, const std::shared_ptr<Tensor>& log_base);

	// Computes the logarithm of a tensor and scalar -> r = log_base(arg)
	static std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& arg, double log_base);

	// Computes the logarithm of a scalar and tensor -> r = log_base(arg)
	static std::shared_ptr<Tensor> log(double arg, const std::shared_ptr<Tensor>& log_base);

	// Computes the natural logarithm of a tensor -> r = ln(t)
	static std::shared_ptr<Tensor> ln(const std::shared_ptr<Tensor>& t);

	// Computes the matrix multiplication of two tensors -> r = a @ b
	static std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	// Computes the convolution of an input and kernels tensor -> r = convolve(input, kernels)
	static std::shared_ptr<Tensor> convolve(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& kernels);

	/* Utilities */

	// Masks the given Q Values based on the given action indices.
	static std::shared_ptr<Tensor> mask_actions(const std::shared_ptr<Tensor>& q_values, const std::vector<int>& actions);

	// Returns the index of the largest value in a tensor.
	static int arg_max(const std::shared_ptr<const Tensor>& t);

	// Computes the sum of a tensor.
	static std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& t);

	// Computes the mean of a tensor.
	static std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& t);

	// Transposes a tensor using the given permutation order.
	static std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& t, const std::vector<int>& axes);

	// Transposes a tensor using the default permutation order - reverse dimension order.
	static std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& t);

	// Broadcasts a tensor into the given target dimensions.
	static std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& t, const std::vector<int>& target_dims);

	// Reshapes a tensor into the given new dimensions.
	static std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& t, const std::vector<int>& new_dims);

	// Flattens the tensor starting at the given axis.
	static std::shared_ptr<Tensor> flatten(const std::shared_ptr<Tensor>& t, int start_axis = 0);

	// Wraps a tensor into a batched tensor with a single input.
	static std::shared_ptr<Tensor> wrap_batch(const std::shared_ptr<Tensor>& t);

	// Clips all the values in a tensor between the given min and max values.
	static std::shared_ptr<Tensor> clip(const std::shared_ptr<Tensor>& t, double min, double max);

	// Creates a dropout mask with the given dimensions.
	static std::shared_ptr<Tensor> get_dense_dropout_mask(const std::vector<int>& dims, double dropout);

	// Creates a spatial dropout mask with the given dimensions.
	static std::shared_ptr<Tensor> get_spatial_dropout_mask(const std::vector<int>& dims, double dropout);

	/* Activation functions */

	// Applies the Rectified Linear Unit function to a tensor.
	static std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t);

	// Applies the Leaky Rectified Linear Unit function to a tensor.
	static std::shared_ptr<Tensor> leaky_relu(const std::shared_ptr<Tensor>& t, double tau);

	// Applies the Sigmoid function to a tensor.
	static std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t);

	// Applies the Tanh function to a tensor.
	static std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t);

	// Applies the Softmax function to a tensor.
	static std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t);

	/* Cost functions - per-element */

	// Computes the Mean Squared Error loss of a tensor based on the given target tensor.
	static std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target);

	// Computes the pseudo-Huber loss of a tensor based on the given target tensor.
	static std::shared_ptr<Tensor> huber(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target, double delta);

	// Computes the Softmax Cross-Entropy loss of a tensor based on the given target tensor (one-hot encoded).
	static std::shared_ptr<Tensor> softmax_cross_entropy(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target);

private:
	/* Internal data */

	// Data vector of the tensor.
	std::vector<double> _data;
	// Gradient vector of the tensor.
	std::vector<double> _grad;
	// Dimensions vector of the tensor.
	std::vector<int> _dimensions;
	// Strides vector of the tensor.
	std::vector<int> _strides;

	/* Parallelization parameters */

	// Threshold for parallelizing matrix multiplication.
	static constexpr long MATMUL_PARALLEL_THRESHOLD = 2'000'000;
	// Threshold for parallelizing convolutions.
	static constexpr long CONV_PARALLEL_THRESHOLD = 2'000'000;

	/* Autograd graph properties */

	// Parent tensors of the tensor.
	std::vector<std::shared_ptr<Tensor>> _parents;
	// Result tensors created from the tensor - for allocation reuse across forward passes.
	std::vector<std::shared_ptr<Tensor>> _results;
	// Index of the current operation in the current autograd graph.
	int _op_index = 0;
	// Backward gradient calculation function of the tensor.
	std::function<void()> _backward = [] {};
	// Global autograd graph generation.
	static int _forward_gen;
	// Most recent generation this tensor participated in.
	int _last_gen = -1;
	// Topography vector.
	std::optional<std::vector<std::shared_ptr<Tensor>>> _topo = {};
	// Topography construction visited hashset.
	std::optional<std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>> _visited = {};

	/* Internal functionality */

	// Computes the strides of the tensor using the given dimensions.
	static std::vector<int> compute_strides(const std::vector<int>& dims);

	// Converts a linear index into indices in the tensor and writes the result into the provided int* array.
	void get_full_indices(int index, int* __restrict indices) const;

	// Prepares the tensor for the next forward pass.
	void prepare_forward();

	// Finalizes the current forward pass for the tensor.
	void finalize_forward();

	// Builds the topography of the current autograd graph.
	static void build_topo(const std::shared_ptr<Tensor>& t, std::vector<std::shared_ptr<Tensor>>& topo,
		std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>& visited);

	// Returns the tensor instance to write the result of the current autograd graph operation into.
	static std::shared_ptr<Tensor> get_result_tensor(const std::shared_ptr<Tensor>& owner, const std::vector<int>& dims, bool requires_grad);
};

/* Free function operators */

// Adds two tensors -> r = a + b
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::add(a, b);
}

// Adds a tensor and scalar -> r = a + b
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::add(a, b);
}

// Adds a scalar and tensor -> r = a + b
inline std::shared_ptr<Tensor> operator+(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::add(b, a); // computative -> a + b = b + a
}

// Subtracts two tensors -> r = a - b
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::sub(a, b);
}

// Subtracts a tensor and scalar -> r = a - b
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::sub(a, b);
}

// Subtracts a scalar and tensor -> r = a - b
inline std::shared_ptr<Tensor> operator-(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::sub(a, b);
}

// Multiplies two tensors -> r = a * b
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::mul(a, b);
}

// Multiplies a tensor and scalar -> r = a * b
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::mul(a, b);
}

// Multiplies a scalar and tensor -> r = a * b
inline std::shared_ptr<Tensor> operator*(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::mul(b, a);
}

// Divides two tensors -> r = a / b
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::div(a, b);
}

// Divides a tensor and scalar -> r = a / b
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, double b)
{
	return Tensor::div(a, b);
}

// Divides a scalar and tensor -> r = a / b
inline std::shared_ptr<Tensor> operator/(double a, const std::shared_ptr<Tensor>& b)
{
	return Tensor::div(a, b);
}