#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"
#include <Windows.h>

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, owner->_dimensions, a->requires_grad || b->requires_grad);

	MathUtils::vector_add(a->_data, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

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

	MathUtils::vector_add(a->_data, b, result->_data);

	if (!inference)
	{
		result->_parents.push_back(a);

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

	MathUtils::vector_sub(a->_data, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

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

	MathUtils::vector_sub(a->_data, b, result->_data);

	if (!inference)
	{
		result->_parents.push_back(a);

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

	MathUtils::vector_sub(a, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(b);

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

	MathUtils::vector_mul(a->_data, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

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

	MathUtils::vector_mul(a->_data, b, result->_data);

	if (!inference)
	{
		result->_parents.push_back(a);

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

	MathUtils::vector_div(a->_data, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

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

	MathUtils::vector_div(a->_data, b, result->_data);

	if (!inference)
	{
		result->_parents.push_back(a);

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

	MathUtils::vector_div(a, b->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(b);

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

	MathUtils::vector_pow(a->_data, exp->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(exp);

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

	MathUtils::vector_pow(a->_data, exp, result->_data);

	if (!inference)
	{
		result->_parents.push_back(a);

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

	MathUtils::vector_pow(a, exp->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(exp);

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

	MathUtils::vector_exp(t->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

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

	MathUtils::vector_log(arg->_data, log_base->_data, result->_data);

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(arg);
		result->_parents.push_back(log_base);

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

	MathUtils::vector_log(arg->_data, log_base, result->_data);

	if (!inference)
	{
		result->_parents.push_back(arg);

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

	MathUtils::vector_log(arg, log_base->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(log_base);

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

	MathUtils::vector_ln(t->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

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

void Tensor::transpose_matrix(const double* __restrict src, double* __restrict dst, int src_off, int dst_off,
	int rows, int cols)
{
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			dst[dst_off + c * rows + r] = src[src_off + r * cols + c]; // switch row and column indices of data
		}
	}
}

void Tensor::compute_row(int i, int n, int p, const double* __restrict a, const double* __restrict b_t,
	double* __restrict r, int a_off, int b_t_off, int r_off)
{
	for (int j = 0; j < p; ++j)
	{
		r[r_off + i * p + j] = MathUtils::vector_dot(a, b_t, a_off + i * n, b_t_off + j * n, n);
	}
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
{
	const int rank = a->rank();
	const int m = a->_dimensions[rank - 2];
	const int n = a->_dimensions[rank - 1];
	const int p = b->_dimensions[b->_dimensions.size() - 1];

	bool b_batched = b->rank() == rank;

	int batch_size = 1;
	for (int i = 0; i < rank - 2; ++i) batch_size *= a->_dimensions[i];
	const int a_mat_size = m * n;
	const int b_mat_size = n * p;
	const int r_mat_size = m * p;

	const int total_rows = batch_size * m;

	std::vector<int> result_dims(a->_dimensions);
	result_dims[rank - 1] = p;

	const auto& owner = a->requires_grad ? a : b;
	auto result = get_result_tensor(owner, result_dims, a->requires_grad || b->requires_grad);

	const bool use_parallel = total_rows > 16 && (long)m * n * p > MATMUL_PARALLEL_THRESHOLD;
	std::vector<double> b_t(b_mat_size * batch_size);
	int b_src_off = 0;
	for (int batch = 0; batch < batch_size; ++batch)
	{
		if (b_batched) b_src_off = batch * b_mat_size;
		transpose_matrix(b->_data.data(), b_t.data(), b_src_off, batch * b_mat_size, n, p);
	}

	#pragma warning(suppress: 6993)
	#pragma omp parallel for if(use_parallel)
	for (int row = 0; row < total_rows; ++row)
	{
		const int batch = row / m;
		const int i = row % m;

		compute_row(i, n, p, a->_data.data(), b_t.data(), result->_data.data(), batch * a_mat_size, batch * b_mat_size, batch * r_mat_size);
	}

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 2);
		result->_parents.push_back(a);
		result->_parents.push_back(b);

		result->_backward = [a, a_mat_size, b, b_mat_size, b_batched, batch_size, m, n, p, result, r_mat_size]()
			{
				if (!a->requires_grad && !b->requires_grad) return;

				const bool par = (long)m * n * p > MATMUL_PARALLEL_THRESHOLD;

				std::vector<double> a_t;
				std::vector<double> d_out_t;

				int b_off = 0;
				for (int batch = 0; batch < batch_size; ++batch)
				{
					const int a_off = batch * a_mat_size;
					if (b_batched) b_off = batch * b_mat_size;
					const int r_off = batch * r_mat_size;

					if (a->requires_grad && b->requires_grad)
					{
						a_t.resize(a_mat_size);
						d_out_t.resize(r_mat_size);

						transpose_matrix(a->_data.data(), a_t.data(), a_off, 0, m, n);
						transpose_matrix(result->_grad.data(), d_out_t.data(), r_off, 0, m, p);

						#pragma omp parallel for if(par)
						for (int i = 0; i < m; ++i)
						{
							for (int k = 0; k < n; ++k)
							{
								a->_grad[a_off + i * n + k] += MathUtils::vector_dot(result->_grad.data(), b->_data.data(),
									r_off + i * p, b_off + k * p, p);
							}
						}

						#pragma omp parallel for if(par && b_batched)
						for (int k = 0; k < n; ++k)
						{
							for (int j = 0; j < p; ++j)
							{
								b->_grad[b_off + k * p + j] += MathUtils::vector_dot(a_t.data(), d_out_t.data(),
									k * m, j * m, m);
							}
						}
					}
					else if (a->requires_grad)
					{
						#pragma omp parallel for if(par)
						for (int i = 0; i < m; ++i)
						{
							for (int k = 0; k < n; ++k)
							{
								a->_grad[a_off + i * n + k] += MathUtils::vector_dot(result->_grad.data(), b->_data.data(),
									r_off + i * p, b_off + k * p, p);
							}
						}
					}
					else if (b->requires_grad)
					{
						a_t.resize(a_mat_size);
						d_out_t.resize(r_mat_size);

						transpose_matrix(a->_data.data(), a_t.data(), a_off, 0, m, n);
						transpose_matrix(result->_grad.data(), d_out_t.data(), r_off, 0, m, p);

						#pragma omp parallel for if(par && b_batched)
						for (int k = 0; k < n; ++k)
						{
							for (int j = 0; j < p; ++j)
							{
								b->_grad[b_off + k * p + j] += MathUtils::vector_dot(a_t.data(), d_out_t.data(),
									k * m, j * m, m);
							}
						}
					}
				}
			};
	}

	return result;
}

void Tensor::compute_output_position(int batch_out_pos, int spatial_rank, int out_spatial_size, int filter_count,
	int kernel_spatial_size, int input_channels, const int* __restrict out_spatial_strides,
	const int* __restrict kernel_spatial_strides, const int* __restrict input_strides,
	const int* __restrict kernel_strides, const int* __restrict result_strides, const double* __restrict input_data,
	const double* __restrict kernel_data, const double* __restrict bias_data, double* __restrict result_data)
{
	const int b = batch_out_pos / out_spatial_size;

	std::vector<int> out_coords(spatial_rank);
	std::vector<int> kernel_coords(spatial_rank);
	std::vector<double> sums(filter_count);

	int result_offset = b * result_strides[0];
	int rem = batch_out_pos % out_spatial_size;
	for (int i = 0; i < spatial_rank; ++i)
	{
		out_coords[i] = rem / out_spatial_strides[i];
		result_offset += out_coords[i] * result_strides[i + 1];
		rem %= out_spatial_strides[i];
	}

	const int input_offset_base = b * input_strides[0];
	const int kernel_offset_base_coeff = kernel_spatial_size * input_channels;

	std::copy_n(bias_data, filter_count, sums.data());

	for (int kp = 0; kp < kernel_spatial_size; ++kp)
	{
		int kernel_offset_base = 0;
		int input_offset = input_offset_base;
		rem = kp;
		for (int i = 0; i < spatial_rank; ++i)
		{
			kernel_coords[i] = rem / kernel_spatial_strides[i];
			kernel_offset_base += kernel_coords[i] * kernel_strides[i + 1];
			input_offset += (out_coords[i] + kernel_coords[i]) * input_strides[i + 1];
			rem %= kernel_spatial_strides[i];
		}

		for (int f = 0; f < filter_count; ++f)
		{
			const int kernel_offset = kernel_offset_base_coeff * f + kernel_offset_base;
			sums[f] += MathUtils::vector_dot(input_data, kernel_data, input_offset, kernel_offset, input_channels);
		}
	}

	std::copy_n(sums.data(), filter_count, result_data + result_offset);
}

void Tensor::compute_kernel_grad(int fkp, int spatial_rank, int batches, int out_spatial_size,
	int kernel_spatial_size, int input_channels, const int* __restrict out_spatial_strides,
	const int* __restrict kernel_spatial_strides, const int* __restrict input_strides,
	const int* __restrict kernel_strides, const int* __restrict result_strides, const double* __restrict input_data,
	double* __restrict kernel_grad, const double* __restrict result_grad)
{
	const int f = fkp / kernel_spatial_size;

	std::vector<int> kernel_coords(spatial_rank);
	std::vector<int> out_coords(spatial_rank);

	int rem = fkp % kernel_spatial_size;
	const int kernel_offset = f * kernel_strides[0] + rem * input_channels;
	for (int i = 0; i < spatial_rank; ++i)
	{
		kernel_coords[i] = rem / kernel_spatial_strides[i];
		rem %= kernel_spatial_strides[i];
	}

	for (int b = 0; b < batches; ++b)
	{
		const int input_offset_base = b * input_strides[0];
		const int result_offset_base = b * result_strides[0] + f;

		for (int op = 0; op < out_spatial_size; ++op)
		{
			int input_offset = input_offset_base;
			int result_offset = result_offset_base;
			rem = op;
			for (int i = 0; i < spatial_rank; ++i)
			{
				out_coords[i] = rem / out_spatial_strides[i];
				input_offset += (out_coords[i] + kernel_coords[i]) * input_strides[i + 1];
				result_offset += out_coords[i] * result_strides[i + 1];
				rem %= out_spatial_strides[i];
			}
			const double d_out = result_grad[result_offset];

			MathUtils::vector_fnmadd(kernel_grad + kernel_offset, input_data + input_offset, d_out, input_channels);
		}
	}
}

void Tensor::compute_input_grad(int batch_in_pos, int spatial_rank, int in_spatial_size, int filter_count,
	int kernel_spatial_size, int input_channels, const int* __restrict in_spatial_strides,
	const int* __restrict kernel_spatial_strides, const int* __restrict out_spatial_dims,
	const int* __restrict input_strides, const int* __restrict kernel_strides, const int* __restrict result_strides,
	double* __restrict input_grad, const double* __restrict kernel_data, const double* __restrict result_grad)
{
	const int b = batch_in_pos / in_spatial_size;

	std::vector<int> in_coords(spatial_rank);
	std::vector<int> kernel_coords(spatial_rank);

	int input_offset = b * input_strides[0];
	int rem = batch_in_pos % in_spatial_size;
	for (int i = 0; i < spatial_rank; ++i)
	{
		in_coords[i] = rem / in_spatial_strides[i];
		input_offset += in_coords[i] * input_strides[i + 1];
		rem %= in_spatial_strides[i];
	}

	const int result_offset_base = b * result_strides[0];
	const int kernel_offset_base_coeff = kernel_spatial_size * input_channels;

	for (int f = 0; f < filter_count; ++f)
	{
		const int kernel_offset_base = f * kernel_offset_base_coeff;
		const int result_offset_filter = result_offset_base + f;

		for (int kp = 0; kp < kernel_spatial_size; ++kp)
		{
			int result_offset = result_offset_filter;
			int kernel_offset = kernel_offset_base;
			bool valid = true;
			rem = kp;
			for (int i = 0; i < spatial_rank; ++i)
			{
				kernel_coords[i] = rem / kernel_spatial_strides[i];
				const int out_coord = in_coords[i] - kernel_coords[i];
				if (out_coord < 0 || out_coord >= out_spatial_dims[i])
				{
					valid = false;
					break;
				}
				result_offset += out_coord * result_strides[i + 1];
				kernel_offset += kernel_coords[i] * kernel_strides[i + 1];
				rem %= kernel_spatial_strides[i];
			}
			if (!valid) continue;

			const double d_out = result_grad[result_offset];

			MathUtils::vector_fmadd(input_grad + input_offset, kernel_data + kernel_offset, d_out, input_channels);
		}
	}
}

std::shared_ptr<Tensor> Tensor::convolve(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& kernels,
	const std::shared_ptr<Tensor>& biases)
{
	const int batches = input->_dimensions[0];
	const int spatial_rank = kernels->rank() - 2;
	const int filter_count = kernels->_dimensions[0];
	const int input_channels = kernels->_dimensions.back();

	std::vector<int> out_spatial_dims(spatial_rank);
	std::vector<int> in_spatial_strides(spatial_rank);
	std::vector<int> out_spatial_strides(spatial_rank);
	std::vector<int> kernel_spatial_strides(spatial_rank);
	std::vector<int> result_dims(input->rank());

	result_dims[0] = batches;
	result_dims.back() = filter_count;
	int out_spatial_size = 1;
	int in_spatial_size = 1;
	int kernel_spatial_size = 1;
	for (int i = 0; i < spatial_rank; ++i)
	{
		out_spatial_dims[i] = input->_dimensions[i + 1] - kernels->_dimensions[i + 1] + 1;
		out_spatial_size *= out_spatial_dims[i];
		in_spatial_size *= input->_dimensions[i + 1];
		kernel_spatial_size *= kernels->_dimensions[i + 1];
		result_dims[i + 1] = out_spatial_dims[i];
	}

	const int kernel_volume_size = kernel_spatial_size * input_channels;

	in_spatial_strides.back() = 1;
	out_spatial_strides.back() = 1;
	kernel_spatial_strides.back() = 1;
	for (int i = spatial_rank - 2; i >= 0; --i)
	{
		in_spatial_strides[i] = in_spatial_strides[i + 1] * input->_dimensions[i + 2];
		out_spatial_strides[i] = out_spatial_strides[i + 1] * out_spatial_dims[i + 1];
		kernel_spatial_strides[i] = kernel_spatial_strides[i + 1] * kernels->_dimensions[i + 2];
	}

	const auto& owner = input->requires_grad ? input : (kernels->requires_grad ? kernels : biases);
	auto result = get_result_tensor(owner, result_dims, input->requires_grad || kernels->requires_grad || biases->requires_grad);

	const bool use_parallel = (long)batches * out_spatial_size * filter_count * kernel_volume_size > CONV_PARALLEL_THRESHOLD;

	#pragma omp parallel for if(use_parallel)
	for (int i = 0; i < batches * out_spatial_size; ++i)
	{
		compute_output_position(i, spatial_rank, out_spatial_size, filter_count, kernel_spatial_size, input_channels,
			out_spatial_strides.data(), kernel_spatial_strides.data(), input->_strides.data(), kernels->_strides.data(),
			result->_strides.data(), input->_data.data(), kernels->_data.data(), biases->_data.data(), result->_data.data());
	}

	if (!inference)
	{
		result->_parents.reserve(result->_parents.size() + 3);
		result->_parents.push_back(input);
		result->_parents.push_back(kernels);
		result->_parents.push_back(biases);

		result->_backward = [batches, spatial_rank, filter_count, input_channels, input, in_spatial_size, in_spatial_strides,
			kernels, kernel_spatial_size, kernel_spatial_strides, kernel_volume_size, biases, result, out_spatial_size,
			out_spatial_dims, out_spatial_strides]()
			{
				const bool par = (long)batches * out_spatial_size * filter_count * kernel_volume_size > CONV_PARALLEL_THRESHOLD;

				if (biases->requires_grad)
				{
					double* const __restrict b_grad_ptr = biases->_grad.data();
					const double* const __restrict r_grad_ptr = result->_grad.data();

					for (int b = 0; b < batches; ++b)
					{
						const int r_off_coeff = b * out_spatial_size;
						for (int s = 0; s < out_spatial_size; ++s)
						{
							const int r_off = (r_off_coeff + s) * filter_count;
							MathUtils::vector_add(b_grad_ptr, r_grad_ptr + r_off, filter_count);
						}
					}
				}

				if (kernels->requires_grad)
				{
					#pragma omp parallel for if(par)
					for (int fkp = 0; fkp < filter_count * kernel_spatial_size; ++fkp)
					{
						compute_kernel_grad(fkp, spatial_rank, batches, out_spatial_size,
							kernel_spatial_size, input_channels, out_spatial_strides.data(), kernel_spatial_strides.data(),
							input->_strides.data(), kernels->_strides.data(), result->_strides.data(), input->_data.data(),
							kernels->_grad.data(), result->_grad.data());
					}
				}

				if (input->requires_grad)
				{
					#pragma omp parallel for if(par)
					for (int batch_in_pos = 0; batch_in_pos < batches * in_spatial_size; ++batch_in_pos)
					{
						compute_input_grad(batch_in_pos, spatial_rank, in_spatial_size, filter_count, kernel_spatial_size,
							input_channels, in_spatial_strides.data(), kernel_spatial_strides.data(), out_spatial_dims.data(),
							input->_strides.data(), kernels->_strides.data(), result->_strides.data(),
							input->_grad.data(), kernels->_data.data(), result->_grad.data());
					}
				}
			};
	}

	return result;
}