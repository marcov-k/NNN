#pragma once

#include "Tensor.h"
#include "Optimizers.h"

extern "C"
{
	__declspec(dllexport) void* tensor_create(const int* dims, int rank, bool requires_grad);

	__declspec(dllexport) void* tensor_create_empty();

	__declspec(dllexport) void* tensor_create_scalar(double value, const int* dims, int rank, bool requires_grad);

	__declspec(dllexport) void* tensor_init_weights(int input_count, int neuron_count);

	__declspec(dllexport) void* tensor_init_biases(int neuron_count);

	__declspec(dllexport) void* tensor_init_kernels(int filter_count, const int* kernel_dims, int kernel_rank, int input_channels);

	__declspec(dllexport) void* tensor_copy(void* handle);

	__declspec(dllexport) void tensor_release(void* handle);

	__declspec(dllexport) int tensor_rank(void* handle);

	__declspec(dllexport) const int* tensor_dims_ptr(void* handle);

	__declspec(dllexport) const int* tensor_strides_ptr(void* handle);

	__declspec(dllexport) int tensor_element_count(void* handle);

	__declspec(dllexport) int tensor_grad_count(void* handle);

	__declspec(dllexport) double* tensor_data_ptr(void* handle);

	__declspec(dllexport) double tensor_get_at(void* handle, int index);

	__declspec(dllexport) void tensor_set_at(void* handle, double value, int index);

	__declspec(dllexport) double tensor_get_at_spatial(void* handle, const int* indices, int rank);

	__declspec(dllexport) void tensor_set_at_spatial(void* handle, double value, const int* indices, int rank);

	__declspec(dllexport) double* tensor_grad_ptr(void* handle);

	__declspec(dllexport) int tensor_linear_index(void* handle, const int* indices, int rank);

	__declspec(dllexport) void tensor_get_full_indices(void* handle, int index, int* out_indices);

	__declspec(dllexport) bool tensor_get_requires_grad(void* handle);

	__declspec(dllexport) void tensor_set_requires_grad(void* handle, bool requires_grad);

	__declspec(dllexport) bool tensor_get_inference();

	__declspec(dllexport) void tensor_set_inference(bool inference);

	__declspec(dllexport) void tensor_begin_forward();

	__declspec(dllexport) void tensor_clear_graph(void* handle);

	__declspec(dllexport) void tensor_backward(void* handle);

	__declspec(dllexport) void* tensor_add(void* handle_a, void* handle_b);

	__declspec(dllexport) void* tensor_add_scalar(void* handle_a, double b);

	__declspec(dllexport) void* tensor_sub(void* handle_a, void* handle_b);

	__declspec(dllexport) void* tensor_sub_scalar(void* handle_a, double b);

	__declspec(dllexport) void* tensor_sub_scalar_left(double a, void* handle_b);

	__declspec(dllexport) void* tensor_mul(void* handle_a, void* handle_b);

	__declspec(dllexport) void* tensor_mul_scalar(void* handle_a, double b);

	__declspec(dllexport) void* tensor_div(void* handle_a, void* handle_b);

	__declspec(dllexport) void* tensor_div_scalar(void* handle_a, double b);

	__declspec(dllexport) void* tensor_div_scalar_left(double a, void* handle_b);

	__declspec(dllexport) void* tensor_pow(void* handle_a, void* handle_exp);

	__declspec(dllexport) void* tensor_pow_scalar(void* handle_a, double exp);

	__declspec(dllexport) void* tensor_pow_scalar_left(double a, void* handle_exp);

	__declspec(dllexport) void* tensor_exp(void* handle);

	__declspec(dllexport) void* tensor_log(void* handle_arg, void* handle_log_base);

	__declspec(dllexport) void* tensor_log_scalar(void* handle_arg, double log_base);

	__declspec(dllexport) void* tensor_log_scalar_left(double arg, void* handle_log_base);

	__declspec(dllexport) void* tensor_ln(void* handle);

	__declspec(dllexport) void* tensor_matmul(void* handle_a, void* handle_b);

	__declspec(dllexport) void* tensor_convolve(void* handle_input, void* handle_kernels, void* handle_biases);

	__declspec(dllexport) void* tensor_mask_actions(void* handle_q_values, const int* actions, int action_count);

	__declspec(dllexport) int tensor_arg_max(void* handle);

	__declspec(dllexport) void* tensor_sum(void* handle);

	__declspec(dllexport) void* tensor_mean(void* handle);

	__declspec(dllexport) void* tensor_transpose(void* handle, const int* axes, int axes_length);

	__declspec(dllexport) void* tensor_transpose_default(void* handle);

	__declspec(dllexport) void* tensor_broadcast(void* handle, const int* target_dims, int target_dims_length);

	__declspec(dllexport) void* tensor_reshape(void* handle, const int* new_dims, int new_dims_length);

	__declspec(dllexport) void* tensor_flatten(void* handle, int start_axis);

	__declspec(dllexport) void* tensor_wrap_batch(void* handle);

	__declspec(dllexport) void* tensor_clip(void* handle, double min, double max);

	__declspec(dllexport) void* tensor_get_dense_dropout_mask(const int* dims, int dims_length, double dropout);

	__declspec(dllexport) void* tensor_get_spatial_dropout_mask(const int* dims, int dims_length, double dropout);

	__declspec(dllexport) void* tensor_relu(void* handle);

	__declspec(dllexport) void* tensor_leaky_relu(void* handle, double tau);

	__declspec(dllexport) void* tensor_sigmoid(void* handle);

	__declspec(dllexport) void* tensor_tanh(void* handle);

	__declspec(dllexport) void* tensor_softmax(void* handle);

	__declspec(dllexport) void* tensor_mse(void* handle_t, void* handle_target);

	__declspec(dllexport) void* tensor_huber(void* handle_t, void* handle_target, double delta);

	__declspec(dllexport) void* tensor_softmax_cross_entropy(void* handle_t, void* handle_target);

	__declspec(dllexport) void optimizers_clip_gradients(void** handles, int para_count, double max_norm);

	__declspec(dllexport) void optimizers_sgd(void* handle_para, double lr);

	__declspec(dllexport) void optimizers_adam(void* handle_para, double lr, int iter, double* m, double* v, int moments_count,
		double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay);
}