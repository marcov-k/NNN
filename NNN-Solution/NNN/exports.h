#pragma once

#include "Models.h"
#include "Optimizers.h"
#include "Tensor.h"

// Exported methods for interop with C#.
extern "C"
{
	/* Initialization and disposal */

	// Creates a new tensor instance with the given dimensions and requires_grad flag.
	__declspec(dllexport) void* tensor_create(const int* dims, int rank, bool requires_grad);

	// Creates a new empty tensor instance.
	__declspec(dllexport) void* tensor_create_empty();

	// Creates a new tensor instance with the given dimensions and requires_grad flag and fills it with a single scalar value.
	__declspec(dllexport) void* tensor_create_scalar(double value, const int* dims, int rank, bool requires_grad);

	// Initializes a new weights tensor instance for a layer with the given input and neuron counts.
	__declspec(dllexport) void* tensor_init_weights(int input_count, int neuron_count);

	// Initializes a new bias tensor instance for a layer with the given neuron count.
	__declspec(dllexport) void* tensor_init_biases(int neuron_count);

	// Initializes a new kernels tensor instance for a layer with the given filter count, kernel dimensions, and input channel count.
	__declspec(dllexport) void* tensor_init_kernels(int filter_count, const int* kernel_dims, int kernel_rank, int input_channels);

	// Creates a copy of the given tensor detached from the existing autograd graph.
	__declspec(dllexport) void* tensor_copy(void* handle);

	// Releases the given tensor - deletes its export handle.
	__declspec(dllexport) void tensor_release(void* handle);

	/* Data access */

	// Returns the rank of the given tensor.
	__declspec(dllexport) int tensor_rank(void* handle);

	// Returns a pointer to the start of the dimensions vector of the given tensor.
	__declspec(dllexport) const int* tensor_dims_ptr(void* handle);

	// Returns a pointer to the start of the strides vector of the given tensor.
	__declspec(dllexport) const int* tensor_strides_ptr(void* handle);

	// Returns the number of elements contained in the given tensor.
	__declspec(dllexport) int tensor_element_count(void* handle);

	// Returns the number of gradient values contained in the given tensor.
	__declspec(dllexport) int tensor_grad_count(void* handle);

	// Returns a pointer to the start of the data vector of the given tensor.
	__declspec(dllexport) double* tensor_data_ptr(void* handle);

	// Returns the value at the given linear index in the given tensor.
	__declspec(dllexport) double tensor_get_at(void* handle, int index);

	// Sets the value at the given linear index in the given tensor.
	__declspec(dllexport) void tensor_set_at(void* handle, double value, int index);

	// Returns the value at the given indices in the given tensor.
	__declspec(dllexport) double tensor_get_at_spatial(void* handle, const int* indices, int rank);

	// Sets the value at the given indices in the given tensor.
	__declspec(dllexport) void tensor_set_at_spatial(void* handle, double value, const int* indices, int rank);

	// Returns a pointer to the start of gradient vector of the given tensor.
	__declspec(dllexport) double* tensor_grad_ptr(void* handle);

	// Converts the given indices into a linear index in the given tensor.
	__declspec(dllexport) int tensor_linear_index(void* handle, const int* indices, int rank);

	// Converts the given linear index into indices in the given tensor and writes the result to the provided array.
	__declspec(dllexport) void tensor_get_full_indices(void* handle, int index, int* out_indices);

	// Returns the requires_grad flag of the given tensor.
	__declspec(dllexport) bool tensor_get_requires_grad(void* handle);

	// Sets the requires_grad flag of the given tensor.
	__declspec(dllexport) void tensor_set_requires_grad(void* handle, bool requires_grad);

	/* Debug flags */

	// Sets the log_debug flag of the C++ DLL.
	__declspec(dllexport) void tensor_set_log_debug(bool log_debug);

	/* Autograd graph */

	// Returns the inference flag of the C++ autograd engine.
	__declspec(dllexport) bool tensor_get_inference();

	// Sets the inference flag of the C++ autograd engine.
	__declspec(dllexport) void tensor_set_inference(bool inference);

	// Begins a new forward pass in the C++ autograd engine.
	__declspec(dllexport) void tensor_begin_forward();

	// Clears the autograd graph connections of the given tensor.
	__declspec(dllexport) void tensor_clear_graph(void* handle);

	// Finalizes the current inference forward pass.
	__declspec(dllexport) void tensor_finalize_inference(void* handle);

	// Triggers the backward gradient calculation for the autograd graph starting at the given tensor.
	__declspec(dllexport) void tensor_backward(void* handle);

	/* Tensor operations */

	// Adds the given tensors -> (a (T) + b (T))
	__declspec(dllexport) void* tensor_add(void* handle_a, void* handle_b);

	// Adds the given tensor and scalar -> (a (T) + b (S))
	__declspec(dllexport) void* tensor_add_scalar(void* handle_a, double b);

	// Subtracts the given tensors -> (a (T) - b (T))
	__declspec(dllexport) void* tensor_sub(void* handle_a, void* handle_b);

	// Subtracts the given tensor and scalar -> (a (T) - b (S))
	__declspec(dllexport) void* tensor_sub_scalar(void* handle_a, double b);

	// Subtracts the given scalar and tensor -> (a (S) - b (T))
	__declspec(dllexport) void* tensor_sub_scalar_left(double a, void* handle_b);

	// Multiplies the given tensors -> (a (T) * b (T))
	__declspec(dllexport) void* tensor_mul(void* handle_a, void* handle_b);

	// Multiplies the given tensor and scalar -> (a (T) * b (S))
	__declspec(dllexport) void* tensor_mul_scalar(void* handle_a, double b);

	// Divides the given tensors -> (a (T) / b (T))
	__declspec(dllexport) void* tensor_div(void* handle_a, void* handle_b);

	// Divides the given tensor and scalar -> (a (T) / b (S))
	__declspec(dllexport) void* tensor_div_scalar(void* handle_a, double b);

	// Divides the given scalar and tensor -> (a (S) / b (T))
	__declspec(dllexport) void* tensor_div_scalar_left(double a, void* handle_b);

	// Raises the given tensor to the given exponent tensor -> (a (T) ^ exp (T))
	__declspec(dllexport) void* tensor_pow(void* handle_a, void* handle_exp);

	// Raises the given tensor to the given exponent scalar -> (a (T) ^ exp (S))
	__declspec(dllexport) void* tensor_pow_scalar(void* handle_a, double exp);

	// Raises the given scalar to the given exponent tensor -> (a (S) ^ exp (T))
	__declspec(dllexport) void* tensor_pow_scalar_left(double a, void* handle_exp);

	// Raises e to the power of the given tensor -> (e ^ t)
	__declspec(dllexport) void* tensor_exp(void* handle);

	// Computes the logarithm with the given tensor base of the given tensor -> (log_baseT(arg (T))
	__declspec(dllexport) void* tensor_log(void* handle_arg, void* handle_log_base);

	// Computes the logarithm with the given scalar base of the given tensor -> (log_base(arg (T))
	__declspec(dllexport) void* tensor_log_scalar(void* handle_arg, double log_base);

	// Computes the logarithm with the given tensor base of the given scalar -> (log_baseT(arg (S))
	__declspec(dllexport) void* tensor_log_scalar_left(double arg, void* handle_log_base);

	// Computes the natural logarithm of the given tensor -> (ln(t))
	__declspec(dllexport) void* tensor_ln(void* handle);

	// Computes the matrix multiplication between the given tensors -> (a @ b)
	__declspec(dllexport) void* tensor_matmul(void* handle_a, void* handle_b);

	// Performs a convolution of the given input and kernels tensors -> convolve(input, kernels)
	__declspec(dllexport) void* tensor_convolve(void* handle_input, void* handle_kernels);

	/* Tensor utilities */

	// Masks the given Q Values tensor based on the given action indices.
	__declspec(dllexport) void* tensor_mask_actions(void* handle_q_values, const int* actions, int action_count);

	// Returns the linear index of the highest value in the given tensor.
	__declspec(dllexport) int tensor_arg_max(void* handle);

	// Computes the sum of the given tensor.
	__declspec(dllexport) void* tensor_sum(void* handle);

	// Computes the mean of the given tensor.
	__declspec(dllexport) void* tensor_mean(void* handle);

	// Transposes the given tensor based on the given axis indices.
	__declspec(dllexport) void* tensor_transpose(void* handle, const int* axes, int axes_length);

	// Transposes the given tensor using the default transpose override.
	__declspec(dllexport) void* tensor_transpose_default(void* handle);

	// Broadcasts the given tensor into the given dimensions.
	__declspec(dllexport) void* tensor_broadcast(void* handle, const int* target_dims, int target_dims_length);

	// Reshapes the given tensor into the given dimensions.
	__declspec(dllexport) void* tensor_reshape(void* handle, const int* new_dims, int new_dims_length);

	// Flattens the dimensions of the given tensor starting at the given axis.
	__declspec(dllexport) void* tensor_flatten(void* handle, int start_axis);

	// Wraps the given tensor into a batch of a single input.
	__declspec(dllexport) void* tensor_wrap_batch(void* handle);

	// Clips the values of the given tensor to within the given min and max bounds.
	__declspec(dllexport) void* tensor_clip(void* handle, double min, double max);

	// Initializes a new mask applying a dense layer dropout to a tensor with the given dimensions.
	__declspec(dllexport) void* tensor_get_dense_dropout_mask(const int* dims, int dims_length, double dropout);

	// Initializes a new mask applying spatial dropout to a tensor with the given dimensions.
	__declspec(dllexport) void* tensor_get_spatial_dropout_mask(const int* dims, int dims_length, double dropout);

	/* Activation functions */

	// Applies the ReLU activation function to the given tensor.
	__declspec(dllexport) void* tensor_relu(void* handle);

	// Applies the Leaky ReLU activation function to the given tensor.
	__declspec(dllexport) void* tensor_leaky_relu(void* handle, double tau);

	// Applies the Sigmoid activation function to the given tensor.
	__declspec(dllexport) void* tensor_sigmoid(void* handle);

	// Applies the Tanh activation function to the given tensor.
	__declspec(dllexport) void* tensor_tanh(void* handle);

	// Applies the Softmax activation function to the given tensor.
	__declspec(dllexport) void* tensor_softmax(void* handle);

	/* Cost functions */

	// Computes the per-element Mean Squared Error loss of the given tensor using the given target tensor.
	__declspec(dllexport) void* tensor_mse(void* handle_t, void* handle_target);

	// Computes the per-element pseudo-Huber loss of the given tensor using the given target tensor.
	__declspec(dllexport) void* tensor_huber(void* handle_t, void* handle_target, double delta);

	// Computes the per-element Softmax Cross Entropy loss of the given tensor using the given target tensor.
	__declspec(dllexport) void* tensor_softmax_cross_entropy(void* handle_t, void* handle_target);

	/* Optimizers */

	// Performs a Stochastic Gradient Descent optimizer step on the given parameter tensor.
	__declspec(dllexport) void optimizers_sgd(void* handle_para, double lr);

	// Performs an Adam optimizer step with the given parameters on the given parameter tensor.
	__declspec(dllexport) void optimizers_adam(void* handle_para, double lr, int iter, double* m, double* v, int moments_count,
		double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay);

	/* Models */

	// Clips the gradients of the given parameter tensors using the given max norm.
	__declspec(dllexport) void models_clip_gradients(void** handles, int para_count, double max_norm);

	// Applies a soft update to the given target model's parameters based on the given agent model's parameters.
	__declspec(dllexport) void models_soft_update(void** handles_agent, void** handles_target, int para_count, double tau,
		double one_minus_tau);
}