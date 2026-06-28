#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

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