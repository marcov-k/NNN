#include "pch.h"
#include "exports.h"

static void* wrap_handle(const std::shared_ptr<Tensor>& t)
{
	return static_cast<void*>(new std::shared_ptr<Tensor>(t));
}

extern "C"
{
	void* tensor_create(const int* dims, int rank, bool requires_grad)
	{
		std::vector<int> dims_vec(dims, dims + rank);
		return wrap_handle(std::make_shared<Tensor>(dims_vec, requires_grad));
	}

	void* tensor_create_scalar(double value, const int* dims, int rank, bool requires_grad)
	{
		std::vector<int> dims_vec(dims, dims + rank);
		return wrap_handle(std::make_shared<Tensor>(value, dims_vec, requires_grad));
	}

	void* tensor_init_weights(int input_count, int neuron_count)
	{
		return wrap_handle(Tensor::init_weights(input_count, neuron_count));
	}

	void* tensor_init_biases(int neuron_count)
	{
		return wrap_handle(Tensor::init_biases(neuron_count));
	}

	void* tensor_init_kernels(int filter_count, const int* kernel_dims, int kernel_rank, int input_channels)
	{
		std::vector<int> dims_vec(kernel_dims, kernel_dims + kernel_rank);
		return wrap_handle(Tensor::init_kernels(filter_count, dims_vec, input_channels));
	}

	void* tensor_copy(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle((*tensor_handle)->copy());
	}

	void tensor_release(void* handle)
	{
		auto* tensor_ptr = static_cast<std::shared_ptr<Tensor>*>(handle);
		delete tensor_ptr;
	}

	int tensor_rank(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->rank();
	}

	const int* tensor_dims_ptr(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->dimensions().data();
	}

	int tensor_element_count(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->element_count();
	}

	int tensor_grad_count(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->grad_count();
	}

	double* tensor_data_ptr(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->mutable_data().data();
	}

	double* tensor_grad_ptr(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->mutable_grad().data();
	}

	bool tensor_get_requires_grad(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return (*tensor_handle)->requires_grad;
	}

	void tensor_set_requires_grad(void* handle, bool requires_grad)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		(*tensor_handle)->requires_grad = requires_grad;
	}

	bool tensor_get_inference()
	{
		return Tensor::inference;
	}

	void tensor_set_inference(bool inference)
	{
		Tensor::inference = inference;
	}

	void tensor_begin_forward()
	{
		Tensor::begin_forward();
	}

	void tensor_clear_graph(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		(*tensor_handle)->clear_graph();
	}

	void tensor_backward(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		(*tensor_handle)->backward();
	}

	void* tensor_add(void* handle_a, void* handle_b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::add(*tensor_handle_a, *tensor_handle_b));
	}

	void* tensor_add_scalar(void* handle_a, double b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		return wrap_handle(Tensor::add(*tensor_handle_a, b));
	}

	void* tensor_sub(void* handle_a, void* handle_b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::sub(*tensor_handle_a, *tensor_handle_b));
	}

	void* tensor_sub_scalar(void* handle_a, double b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		return wrap_handle(Tensor::sub(*tensor_handle_a, b));
	}

	void* tensor_sub_scalar_left(double a, void* handle_b)
	{
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::sub(a, *tensor_handle_b));
	}

	void* tensor_mul(void* handle_a, void* handle_b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::mul(*tensor_handle_a, *tensor_handle_b));
	}

	void* tensor_mul_scalar(void* handle_a, double b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		return wrap_handle(Tensor::mul(*tensor_handle_a, b));
	}

	void* tensor_div(void* handle_a, void* handle_b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::div(*tensor_handle_a, *tensor_handle_b));
	}

	void* tensor_div_scalar(void* handle_a, double b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		return wrap_handle(Tensor::div(*tensor_handle_a, b));
	}

	void* tensor_div_scalar_left(double a, void* handle_b)
	{
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::div(a, *tensor_handle_b));
	}

	void* tensor_pow(void* handle_a, void* handle_exp)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_exp = static_cast<std::shared_ptr<Tensor>*>(handle_exp);
		return wrap_handle(Tensor::pow(*tensor_handle_a, *tensor_handle_exp));
	}

	void* tensor_pow_scalar(void* handle_a, double exp)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		return wrap_handle(Tensor::pow(*tensor_handle_a, exp));
	}

	void* tensor_pow_scalar_left(double a, void* handle_exp)
	{
		auto* tensor_handle_exp = static_cast<std::shared_ptr<Tensor>*>(handle_exp);
		return wrap_handle(Tensor::pow(a, *tensor_handle_exp));
	}

	void* tensor_exp(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::exp(*tensor_handle));
	}

	void* tensor_log(void* handle_arg, void* handle_log_base)
	{
		auto* tensor_handle_arg = static_cast<std::shared_ptr<Tensor>*>(handle_arg);
		auto* tensor_handle_log_base = static_cast<std::shared_ptr<Tensor>*>(handle_log_base);
		return wrap_handle(Tensor::log(*tensor_handle_arg, *tensor_handle_log_base));
	}

	void* tensor_log_scalar(void* handle_arg, double log_base)
	{
		auto* tensor_handle_arg = static_cast<std::shared_ptr<Tensor>*>(handle_arg);
		return wrap_handle(Tensor::log(*tensor_handle_arg, log_base));
	}

	void* tensor_log_scalar_left(double arg, void* handle_log_base)
	{
		auto* tensor_handle_log_base = static_cast<std::shared_ptr<Tensor>*>(handle_log_base);
		return wrap_handle(Tensor::log(arg, *tensor_handle_log_base));
	}

	void* tensor_matmul(void* handle_a, void* handle_b)
	{
		auto* tensor_handle_a = static_cast<std::shared_ptr<Tensor>*>(handle_a);
		auto* tensor_handle_b = static_cast<std::shared_ptr<Tensor>*>(handle_b);
		return wrap_handle(Tensor::matmul(*tensor_handle_a, *tensor_handle_b));
	}

	void* tensor_convolve(void* handle_input, void* handle_kernels, void* handle_biases)
	{
		auto* tensor_handle_input = static_cast<std::shared_ptr<Tensor>*>(handle_input);
		auto* tensor_handle_kernels = static_cast<std::shared_ptr<Tensor>*>(handle_kernels);
		auto* tensor_handle_biases = static_cast<std::shared_ptr<Tensor>*>(handle_biases);
		return wrap_handle(Tensor::convolve(*tensor_handle_input, *tensor_handle_kernels, *tensor_handle_biases));
	}

	void* tensor_mask_actions(void* handle_q_values, const int* actions, int action_count)
	{
		auto* tensor_handle_q_values = static_cast<std::shared_ptr<Tensor>*>(handle_q_values);
		std::vector<int> actions_vec(actions, actions + action_count);
		return wrap_handle(Tensor::mask_actions(*tensor_handle_q_values, actions_vec));
	}

	int tensor_arg_max(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return Tensor::arg_max(*tensor_handle);
	}

	void* tensor_sum(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::sum(*tensor_handle));
	}

	void* tensor_mean(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::mean(*tensor_handle));
	}

	void* tensor_transpose(void* handle, const int* axes, int axes_length)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		std::vector<int> axes_vec(axes, axes + axes_length);
		return wrap_handle(Tensor::transpose(*tensor_handle, axes_vec));
	}

	void* tensor_transpose_default(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::transpose(*tensor_handle));
	}

	void* tensor_broadcast(void* handle, const int* target_dims, int target_dims_length)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		std::vector<int> target_dims_vec(target_dims, target_dims + target_dims_length);
		return wrap_handle(Tensor::broadcast(*tensor_handle, target_dims_vec));
	}

	void* tensor_reshape(void* handle, const int* new_dims, int new_dims_length)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		std::vector<int> new_dims_vec(new_dims, new_dims + new_dims_length);
		return wrap_handle(Tensor::reshape(*tensor_handle, new_dims_vec));
	}

	void* tensor_flatten(void* handle, int start_axis)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::flatten(*tensor_handle, start_axis));
	}

	void* tensor_wrap_batch(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::wrap_batch(*tensor_handle));
	}

	void* tensor_clip(void* handle, double min, double max)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::clip(*tensor_handle, min, max));
	}

	void* tensor_get_dense_dropout_mask(const int* dims, int dims_length, double dropout)
	{
		std::vector<int> dims_vec(dims, dims + dims_length);
		return wrap_handle(Tensor::get_dense_dropout_mask(dims_vec, dropout));
	}

	void* tensor_get_spatial_dropout_mask(const int* dims, int dims_length, double dropout)
	{
		std::vector<int> dims_vec(dims, dims + dims_length);
		return wrap_handle(Tensor::get_spatial_dropout_mask(dims_vec, dropout));
	}

	void* tensor_relu(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::relu(*tensor_handle));
	}

	void* tensor_leaky_relu(void* handle, double tau)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::leaky_relu(*tensor_handle, tau));
	}

	void* tensor_sigmoid(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::sigmoid(*tensor_handle));
	}

	void* tensor_tanh(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::tanh(*tensor_handle));
	}

	void* tensor_softmax(void* handle)
	{
		auto* tensor_handle = static_cast<std::shared_ptr<Tensor>*>(handle);
		return wrap_handle(Tensor::softmax(*tensor_handle));
	}

	void* tensor_mse(void* handle_t, void* handle_target)
	{
		auto* tensor_handle_t = static_cast<std::shared_ptr<Tensor>*>(handle_t);
		auto* tensor_handle_target = static_cast<std::shared_ptr<Tensor>*>(handle_target);
		return wrap_handle(Tensor::mse(*tensor_handle_t, *tensor_handle_target));
	}

	void* tensor_huber(void* handle_t, void* handle_target, double delta)
	{
		auto* tensor_handle_t = static_cast<std::shared_ptr<Tensor>*>(handle_t);
		auto* tensor_handle_target = static_cast<std::shared_ptr<Tensor>*>(handle_target);
		return wrap_handle(Tensor::huber(*tensor_handle_t, *tensor_handle_target, delta));
	}

	void* tensor_softmax_cross_entropy(void* handle_t, void* handle_target)
	{
		auto* tensor_handle_t = static_cast<std::shared_ptr<Tensor>*>(handle_t);
		auto* tensor_handle_target = static_cast<std::shared_ptr<Tensor>*>(handle_target);
		return wrap_handle(Tensor::softmax_cross_entropy(*tensor_handle_t, *tensor_handle_target));
	}
}