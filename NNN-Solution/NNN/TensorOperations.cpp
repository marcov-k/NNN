#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

/* Tensor operations - autograd graph connected */

// Adds two tensors -> r = a + b
std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || b->requires_grad);

	// Compute addition result
	MathUtils::vector_add(a->_data, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/da = 1; dr/db = 1
		result->_backward = [a, b, result]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				if (a->requires_grad) MathUtils::vector_add(a->_grad, result->_grad);
				if (b->requires_grad) MathUtils::vector_add(b->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& a, double b)
{
	auto result = get_result_tensor(a, a->_dimensions, a->requires_grad);

	// Compute addition result
	MathUtils::vector_add(a->_data, b, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(a);

		// Gradient calculation function -> dr/da = 1
		result->_backward = [a, result]()
			{
				if (!a->requires_grad) return;

				MathUtils::vector_add(a->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || b->requires_grad);

	// Compute subtraction result
	MathUtils::vector_sub(a->_data, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/da = 1; dr/db = -1
		result->_backward = [a, b, result]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				if (a->requires_grad) MathUtils::vector_add(a->_grad, result->_grad);
				if (b->requires_grad) MathUtils::vector_sub(b->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor>& a, double b)
{
	auto result = get_result_tensor(a, a->_dimensions, a->requires_grad);

	// Compute subtraction result
	MathUtils::vector_sub(a->_data, b, result->_data);

	// Connect result tensor to autograd graph
	if (!inference)
	{
		result->_parents.push_back(a);

		// Gradient calculation function -> dr/da = 1
		result->_backward = [a, result]()
			{
				if (!a->requires_grad) return;

				MathUtils::vector_add(a->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::sub(double a, const std::shared_ptr<Tensor>& b)
{
	auto result = get_result_tensor(b, b->_dimensions, b->requires_grad);

	// Compute subtraction result
	MathUtils::vector_sub(a, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/db = -1
		result->_backward = [b, result]()
			{
				if (!b->requires_grad) return;

				MathUtils::vector_sub(b->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || b->requires_grad);

	// Compute multiplication result
	MathUtils::vector_mul(a->_data, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/da = b; dr/db = 1
		result->_backward = [a, b, result]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				if (a->requires_grad) MathUtils::vector_fmadd(a->_grad, b->_data, result->_grad);
				if (b->requires_grad) MathUtils::vector_fmadd(b->_grad, a->_data, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor>& a, double b)
{
	auto result = get_result_tensor(a, a->_dimensions, a->requires_grad);

	// Compute multiplication result
	MathUtils::vector_mul(a->_data, b, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(a);

		// Gradient calculation function -> dr/da = b
		result->_backward = [a, b, result]()
			{
				if (!a->requires_grad) return;

				MathUtils::vector_fmadd(a->_grad, result->_grad, b);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || b->requires_grad);

	// Compute division result
	MathUtils::vector_div(a->_data, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/da = 1 / b; dr/db = a / b^2
		result->_backward = [a, b, result]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				if (a->requires_grad)
				{
					MathUtils::vector_div(result->_grad, b->_data, scratch1);
					MathUtils::vector_add(a->_grad, scratch1);
				}
				if (b->requires_grad)
				{
					MathUtils::vector_sq(b->_data, scratch1);
					MathUtils::vector_div(a->_data, scratch1, scratch2);
					MathUtils::vector_fnmadd(b->_grad, scratch2, result->_grad);
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor>& a, double b)
{
	auto result = get_result_tensor(a, a->_dimensions, a->requires_grad);

	// Compute division result
	MathUtils::vector_div(a->_data, b, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(a);

		// Gradient calculation function -> dr/da = 1 / b
		const double recip_b = 1.0 / b;
		result->_backward = [a, recip_b, result]()
			{
				if (!a->requires_grad) return;

				MathUtils::vector_fmadd(a->_grad, result->_grad, recip_b);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::div(double a, const std::shared_ptr<Tensor>& b)
{
	auto result = get_result_tensor(b, b->_dimensions, b->requires_grad);

	// Compute division result
	MathUtils::vector_div(a, b->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(b);

		// Gradient calculation function -> dr/db = a / b^2
		result->_backward = [a, b, result]()
			{
				if (!b->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				MathUtils::vector_sq(b->_data, scratch1);
				MathUtils::vector_div(a, scratch1, scratch2);
				MathUtils::vector_fnmadd(b->_grad, scratch2, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::pow(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& exp)
{
	const auto& owner = a->requires_grad ? a : exp;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || exp->requires_grad);

	// Compute exponentiation result
	MathUtils::vector_pow(a->_data, exp->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(exp);

		// Gradient calculation function -> dr/da = exp * a^(exp - 1); dr/dexp = a^exp * ln(a)
		result->_backward = [a, exp, result]()
			{
				if (!a->requires_grad && !exp->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				if (a->requires_grad)
				{
					MathUtils::vector_sub(exp->_data, 1.0, scratch1);
					MathUtils::vector_pow(a->_data, scratch1, scratch2);
					MathUtils::vector_mul(scratch2, exp->_data);
					MathUtils::vector_fmadd(a->_grad, scratch2, result->_grad);
				}
				if (exp->requires_grad)
				{
					MathUtils::vector_ln(a->_data, scratch1);
					MathUtils::vector_mul(scratch1, result->_data);
					MathUtils::vector_fmadd(exp->_grad, scratch1, result->_grad);
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::pow(const std::shared_ptr<Tensor>& a, double exp)
{
	auto result = get_result_tensor(a, a->_dimensions, a->requires_grad);

	// Compute exponentiation result
	MathUtils::vector_pow(a->_data, exp, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(a);

		// Gradient calculation function -> dr/da = exp * a^(exp - 1)
		const double exp_sub = exp - 1.0;
		result->_backward = [a, exp, exp_sub, result]()
			{
				if (!a->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);

				MathUtils::vector_pow(a->_data, exp_sub, scratch1);
				MathUtils::vector_mul(scratch1, exp);
				MathUtils::vector_fmadd(a->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::pow(double a, const std::shared_ptr<Tensor>& exp)
{
	auto result = get_result_tensor(exp, exp->_dimensions, exp->requires_grad);

	// Compute exponentiation result
	MathUtils::vector_pow(a, exp->_data, result->_data);

	// Connect result tensor to autograd graph
	if (!inference)
	{
		result->_parents.push_back(exp);

		// Gradient calculation function -> dr/dexp = a^exp * ln(a)
		const double a_ln = std::log(a);
		result->_backward = [a_ln, exp, result]()
			{
				if (!exp->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);

				MathUtils::vector_mul(result->_data, a_ln, scratch1);
				MathUtils::vector_fmadd(exp->_grad, scratch1, result->_grad);
			};
	}
	
	return result;
}

std::shared_ptr<Tensor> Tensor::exp(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Compute natural exponentiation result
	MathUtils::vector_exp(t->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = e^t
		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				MathUtils::vector_fmadd(t->_grad, result->_data, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::log(const std::shared_ptr<Tensor>& arg, const std::shared_ptr<Tensor>& log_base)
{
	const auto& owner = arg->requires_grad ? arg : log_base;
	auto result = get_result_tensor(owner, owner->_dimensions, arg->requires_grad || log_base->requires_grad);

	// Compute logarithm result
	MathUtils::vector_log(arg->_data, log_base->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(arg);
		result->_parents.push_back(log_base);

		// Gradient calculation function -> dr/darg = 1 / (arg * ln(base)); dr/dbase = ln(arg) / (base * ln^2(base))
		result->_backward = [arg, log_base, result]()
			{
				if (!arg->requires_grad && !log_base->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;
				thread_local std::vector<double> scratch3;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);
				scratch3.resize(element_count);

				MathUtils::vector_ln(log_base->_data, scratch1);
				if (arg->requires_grad)
				{
					MathUtils::vector_mul(arg->_data, scratch1, scratch2);
					MathUtils::vector_div(1.0, scratch2, scratch3);
					MathUtils::vector_fmadd(arg->_grad, scratch3, result->_grad);
				}
				if (log_base->requires_grad)
				{
					MathUtils::vector_sq(scratch1, scratch2);
					MathUtils::vector_mul(scratch2, log_base->_data);
					MathUtils::vector_ln(arg->_data, scratch3);
					MathUtils::vector_div(scratch3, scratch2);
					MathUtils::vector_fnmadd(log_base->_grad, scratch3, result->_grad);
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::log(const std::shared_ptr<Tensor>& arg, double log_base)
{
	auto result = get_result_tensor(arg, arg->_dimensions, arg->requires_grad);

	// Compute logarithm result
	MathUtils::vector_log(arg->_data, log_base, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(arg);

		// Gradient calculation function -> dr/darg = 1 / (arg * ln(arg))
		const double base_ln = std::log(log_base);
		result->_backward = [arg, base_ln, result]()
			{
				if (!arg->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				MathUtils::vector_mul(arg->_data, base_ln, scratch1);
				MathUtils::vector_div(1.0, scratch1, scratch2);
				MathUtils::vector_fmadd(arg->_grad, scratch2, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::log(double arg, const std::shared_ptr<Tensor>& log_base)
{
	auto result = get_result_tensor(log_base, log_base->_dimensions, log_base->requires_grad);

	// Compute logarithm result
	MathUtils::vector_log(arg, log_base->_data, result->_data);

	// Connect result tensor to autograd graph
	if (!inference)
	{
		result->_parents.push_back(log_base);

		// Gradient calculation function -> dr/dbase = ln(arg) / (base * ln^2(base))
		const double arg_ln = std::log(arg);
		result->_backward = [arg_ln, log_base, result]()
			{
				if (!log_base->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				MathUtils::vector_ln(log_base->_data, scratch1);
				MathUtils::vector_sq(scratch1);
				MathUtils::vector_mul(scratch1, log_base->_data);
				MathUtils::vector_div(arg_ln, scratch1, scratch2);
				MathUtils::vector_fnmadd(log_base->_grad, scratch2, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::ln(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Compute natural logarithm result
	MathUtils::vector_ln(t->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = 1 / t
		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_div(1.0, t->_data, scratch1);
				MathUtils::vector_fmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	// Compute matrix multiplication geometry

	const int rank = a->rank();
	const size_t m = a->_dimensions[rank - 2];
	const size_t n = a->_dimensions[rank - 1];
	const size_t p = b->_dimensions[b->_dimensions.size() - 1];

	const bool b_batched = b->rank() == rank && std::equal(a->_dimensions.begin(), a->_dimensions.end() - 2, b->_dimensions.begin());

	size_t batch_size = 1;
	for (int i = 0; i < rank - 2; ++i) batch_size *= a->_dimensions[i];
	const size_t a_mat_size = m * n;
	const size_t b_mat_size = n * p;
	const size_t r_mat_size = m * p;

	const size_t b_batch_stride = b_batched ? b_mat_size : 0;

	const size_t total_rows = batch_size * m;

	std::vector<int> result_dims(a->_dimensions);
	result_dims[rank - 1] = (int)p;

	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, result_dims, a->requires_grad || b->requires_grad);

	const bool use_parallel = total_rows > 16 && (long)total_rows * n * p > MATMUL_PARALLEL_THRESHOLD;

	// Transpose b
	auto b_t = std::make_shared<std::vector<double>>(b_batched ? b_mat_size * batch_size : b_mat_size);
	if (b_batched)
	{
		for (size_t batch = 0; batch < batch_size; ++batch)
		{
			const size_t b_batch_off = batch * b_mat_size;
			MathUtils::transpose_matrix(b->_data.data(), b_t->data(), b_batch_off, b_batch_off, n, p);
		}
	}
	else
	{
		MathUtils::transpose_matrix(b->_data.data(), b_t->data(), 0, 0, n, p);
	}

	// Compute matrix multiplication result per batch
	MathUtils::matmul_raw(a->_data.data(), b_t->data(), result->_data.data(), batch_size, m, n, p, a_mat_size,
		b_batch_stride, r_mat_size, 0, 0, 0, use_parallel, false);

	// Connect result to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		// Gradient calculation function -> grad_a = grad_r @ b_T; grad_b = a_T @ grad_r
		result->_backward = [a, a_mat_size, b, b_t, b_mat_size, b_batched, b_batch_stride, total_rows,
			batch_size, m, n, p, result, r_mat_size, use_parallel]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				thread_local std::vector<double> d_r_t;
				thread_local std::vector<double> a_t;

				if (a->requires_grad)
				{
					MathUtils::matmul_raw(result->_grad.data(), b->_data.data(), a->_grad.data(), batch_size, m, p, n,
						r_mat_size, b_batch_stride, a_mat_size, 0, 0, 0, use_parallel, true);
				}

				if (b->requires_grad)
				{
					a_t.resize(a_mat_size * batch_size);
					d_r_t.resize(r_mat_size * batch_size);

					for (size_t batch = 0; batch < batch_size; ++batch)
					{
						const size_t a_batch_off = batch * a_mat_size;
						const size_t r_batch_off = batch * r_mat_size;
						MathUtils::transpose_matrix(a->_data.data(), a_t.data(), a_batch_off, a_batch_off, m, n);
						MathUtils::transpose_matrix(result->_grad.data(), d_r_t.data(), r_batch_off, r_batch_off, m, p);
					}

					if (b_batched)
					{
						MathUtils::matmul_raw(a_t.data(), d_r_t.data(), b->_grad.data(), batch_size, n, m, p, a_mat_size,
							r_mat_size, b_mat_size, 0, 0, 0, use_parallel, true);
					}
					else
					{
						MathUtils::matmul_reduce_raw(a_t.data(), d_r_t.data(), b->_grad.data(), batch_size,
							n, m, p, a_mat_size, r_mat_size, 0, 0, 0, use_parallel, true);
					}
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::convolve(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& kernels)
{
	// Compute convolution geometry
	ConvGeometry g;
	g.batches = input->_dimensions[0];
	g.spatial_rank = kernels->rank() - 2;
	g.input_channels = kernels->_dimensions.back();
	g.filter_count = kernels->_dimensions[0];

	g.input_dims = input->_dimensions;
	g.input_strides = input->_strides;
	g.kernel_dims = kernels->_dimensions;
	g.out_dims.resize(g.spatial_rank);

	g.out_spatial_size = 1;
	g.kernel_spatial_size = 1;
	for (size_t i = 0; i < g.spatial_rank; ++i)
	{
		g.out_dims[i] = input->_dimensions[i + 1] - kernels->_dimensions[i + 1] + 1;
		g.out_spatial_size *= g.out_dims[i];
		g.kernel_spatial_size *= kernels->_dimensions[i + 1];
	}

	g.out_spatial_strides.resize(g.spatial_rank);
	g.out_spatial_strides.back() = 1;
	g.kernel_spatial_strides.resize(g.spatial_rank);
	g.kernel_spatial_strides.back() = 1;
	for (int i = (int)g.spatial_rank - 2; i >= 0; --i)
	{
		g.out_spatial_strides[i] = g.out_spatial_strides[i + 1] * g.out_dims[i + 1];
		g.kernel_spatial_strides[i] = g.kernel_spatial_strides[i + 1] * g.kernel_dims[i + 1];
	}

	g.kernel_volume_size = g.kernel_spatial_size * g.input_channels;

	g.im2col_rows = g.batches * g.out_spatial_size;
	g.im2col_cols = g.kernel_volume_size;

	g.input_kernel_offset.resize(g.kernel_volume_size);
	g.kernel_kernel_offset.resize(g.kernel_volume_size);
	for (size_t k = 0; k < g.kernel_volume_size; ++k)
	{
		const size_t spatial_k = k / g.input_channels;
		const size_t c = k % g.input_channels;

		size_t input_offset = 0;
		size_t kernel_offset = 0;
		for (size_t d = 0; d < g.spatial_rank; ++d)
		{
			size_t coord = (spatial_k / g.kernel_spatial_strides[d]) % g.kernel_dims[d + 1];
			input_offset += coord * g.input_strides[d + 1];
			kernel_offset += coord * kernels->_strides[d + 1];
		}

		g.input_kernel_offset[k] = input_offset + c;
		g.kernel_kernel_offset[k] = kernel_offset + c;
	}

	// Compute result dimensions
	thread_local std::vector<int> result_dims;
	result_dims.resize(g.spatial_rank + 2);
	result_dims[0] = (int)g.batches;
	for (size_t d = 0; d < g.spatial_rank; ++d) result_dims[d + 1] = (int)g.out_dims[d];
	result_dims.back() = (int)g.filter_count;

	const auto& owner = input->requires_grad ? input : kernels;
	auto result = get_result_tensor(owner, result_dims, input->requires_grad || kernels->requires_grad);

	auto kernels_mat = std::make_shared<std::vector<double>>(kernels->element_count());
	auto input_col = std::make_shared<std::vector<double>>(g.im2col_rows * g.im2col_cols);

	const bool use_parallel = (long)g.batches * g.out_spatial_size * g.filter_count * g.kernel_volume_size > CONV_PARALLEL_THRESHOLD;

	// Compute convolution - rotate kernels + im2col + matmul
	MathUtils::kernels2matmul(kernels->_data.data(), g, kernels_mat->data());
	MathUtils::im2col(input->_data.data(), g, input_col->data(), use_parallel);

	MathUtils::matmul_raw(input_col->data(), kernels_mat->data(), result->_data.data(), 1, g.im2col_rows, g.im2col_cols,
		g.filter_count, 0, 0, 0, 0, 0, 0, use_parallel, false);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(input);
		result->_parents.push_back(kernels);

		// Gradient calculation function -> grad_input = convolve(grad_r, rotated kernels); grad_kernels = convolve(input, rotated grad_r)
		result->_backward = [input, input_col, kernels, kernels_mat, result, g, use_parallel]()
			{
				// Compute grad_input = convolve(grad_r, rotated kernels)
				if (input->requires_grad)
				{
					thread_local std::vector<double> d_col;
					thread_local std::vector<double> kernels_mat_t;

					d_col.assign(g.im2col_rows * g.im2col_cols, 0.0);
					kernels_mat_t.resize(kernels_mat->size());

					MathUtils::transpose_matrix(kernels_mat->data(), kernels_mat_t.data(), 0, 0, g.filter_count, g.im2col_cols);

					MathUtils::matmul_raw(result->_grad.data(), kernels_mat_t.data(), d_col.data(), 1, g.im2col_rows,
						g.filter_count, g.im2col_cols, 0, 0, 0, 0, 0, 0, use_parallel, false);

					MathUtils::col2im(d_col.data(), g, input->_grad.data(), use_parallel);
				}

				// Compute grad_kernels = convolve(input, rotated grad_r)
				if (kernels->requires_grad)
				{
					thread_local std::vector<double> d_out_t;
					thread_local std::vector<double> input_col_t;
					thread_local std::vector<double> d_kernels_ft;

					d_out_t.resize(g.filter_count * g.im2col_rows);
					input_col_t.resize(input_col->size());
					d_kernels_ft.resize(kernels->element_count());

					MathUtils::transpose_matrix(result->_grad.data(), d_out_t.data(), 0, 0, g.im2col_rows, g.filter_count);

					MathUtils::transpose_matrix(input_col->data(), input_col_t.data(), 0, 0, g.im2col_rows, g.im2col_cols);
					
					MathUtils::matmul_raw(d_out_t.data(), input_col_t.data(), d_kernels_ft.data(), 1, g.filter_count,
						g.im2col_rows, g.im2col_cols, 0, 0, 0, 0, 0, 0, use_parallel, false);

					MathUtils::matmul2kernels(d_kernels_ft.data(), g, kernels->_grad.data(), true);
				}
			};
	}

	return result;
}