#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

std::shared_ptr<Tensor> Tensor::relu(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	MathUtils::vector_max(t->_data, 0.0, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				const size_t n = result->element_count();

				const double* const __restrict p_tv = t->_data.data();
				double* const __restrict p_tg = t->_grad.data();
				const double* const __restrict p_rg = result->_grad.data();
				const __m256d reg_0 = _mm256_setzero_pd();

				size_t i = 0;
				for (; i + 8 <= n; i += 8)
				{
					__m256d reg_tv0 = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tv1 = _mm256_loadu_pd(&p_tv[i + 4]);

					__m256d reg_tg0 = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_tg1 = _mm256_loadu_pd(&p_tg[i + 4]);

					__m256d reg_rg0 = _mm256_loadu_pd(&p_rg[i]);
					__m256d reg_rg1 = _mm256_loadu_pd(&p_rg[i + 4]);

					__m256d mask0 = _mm256_cmp_pd(reg_tv0, reg_0, _CMP_GT_OS);
					__m256d mask1 = _mm256_cmp_pd(reg_tv1, reg_0, _CMP_GT_OS);

					__m256d grad_add0 = _mm256_blendv_pd(reg_0, reg_rg0, mask0);
					__m256d grad_add1 = _mm256_blendv_pd(reg_0, reg_rg1, mask1);

					__m256d grad0 = _mm256_add_pd(reg_tg0, grad_add0);
					__m256d grad1 = _mm256_add_pd(reg_tg1, grad_add1);

					_mm256_storeu_pd(&p_tg[i], grad0);
					_mm256_storeu_pd(&p_tg[i + 4], grad1);
				}

				for (; i + 4 <= n; i += 4)
				{
					__m256d reg_tv = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tg = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_rg = _mm256_loadu_pd(&p_rg[i]);

					__m256d mask = _mm256_cmp_pd(reg_tv, reg_0, _CMP_GT_OS);
					__m256d grad_add = _mm256_blendv_pd(reg_0, reg_rg, mask);
					__m256d grad = _mm256_add_pd(reg_tg, grad_add);
					_mm256_storeu_pd(&p_tg[i], grad);
				}

				for (; i < n; ++i)
				{
					if (p_tv[i] > 0.0)
					{
						p_tg[i] += p_rg[i];
					}
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::leaky_relu(const std::shared_ptr<Tensor>& t, double tau)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	result->_data = t->_data;

	const size_t n = result->element_count();

	const __m256d reg_tau = _mm256_set1_pd(tau);
	double* const __restrict p_r = result->_data.data();
	const __m256d reg_0 = _mm256_setzero_pd();

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_r0 = _mm256_loadu_pd(&p_r[i]);
		__m256d reg_r1 = _mm256_loadu_pd(&p_r[i + 4]);

		__m256d mask0 = _mm256_cmp_pd(reg_r0, reg_0, _CMP_LE_OS);
		__m256d mask1 = _mm256_cmp_pd(reg_r1, reg_0, _CMP_LE_OS);

		__m256d res0 = _mm256_blendv_pd(reg_r0, _mm256_mul_pd(reg_r0, reg_tau), mask0);
		__m256d res1 = _mm256_blendv_pd(reg_r1, _mm256_mul_pd(reg_r1, reg_tau), mask1);

		_mm256_storeu_pd(&p_r[i], res0);
		_mm256_storeu_pd(&p_r[i + 4], res1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_r = _mm256_loadu_pd(&p_r[i]);
		__m256d mask = _mm256_cmp_pd(reg_r, reg_0, _CMP_LE_OS);
		__m256d res = _mm256_blendv_pd(reg_r, _mm256_mul_pd(reg_r, reg_tau), mask);
		_mm256_storeu_pd(&p_r[i], res);
	}
	
	for (; i < n; ++i)
	{
		if (p_r[i] <= 0.0)
		{
			p_r[i] *= tau;
		}
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, tau, result, n]()
			{
				const double* const __restrict p_tv = t->_data.data();
				double* const __restrict p_tg = t->_grad.data();
				const __m256d reg_tau = _mm256_set1_pd(tau);
				const double* const __restrict p_rg = result->_grad.data();
				const __m256d reg_0 = _mm256_setzero_pd();
				const __m256d reg_1 = _mm256_set1_pd(1.0);

				size_t i = 0;
				for (; i + 8 <= n; i += 8)
				{
					__m256d reg_tv0 = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tv1 = _mm256_loadu_pd(&p_tv[i + 4]);

					__m256d reg_tg0 = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_tg1 = _mm256_loadu_pd(&p_tg[i + 4]);

					__m256d reg_rg0 = _mm256_loadu_pd(&p_rg[i]);
					__m256d reg_rg1 = _mm256_loadu_pd(&p_rg[i + 4]);

					__m256d mask0 = _mm256_cmp_pd(reg_tv0, reg_0, _CMP_GT_OS);
					__m256d mask1 = _mm256_cmp_pd(reg_tv1, reg_0, _CMP_GT_OS);

					__m256d blend0 = _mm256_blendv_pd(reg_tau, reg_1, mask0);
					__m256d blend1 = _mm256_blendv_pd(reg_tau, reg_1, mask1);

					__m256d grad0 = _mm256_fmadd_pd(blend0, reg_rg0, reg_tg0);
					__m256d grad1 = _mm256_fmadd_pd(blend1, reg_rg1, reg_tg1);

					_mm256_storeu_pd(&p_tg[i], grad0);
					_mm256_storeu_pd(&p_tg[i + 4], grad1);
				}

				for (; i + 4 <= n; i += 4)
				{
					__m256d reg_tv = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tg = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_rg = _mm256_loadu_pd(&p_rg[i]);

					__m256d mask = _mm256_cmp_pd(reg_tv, reg_0, _CMP_GT_OS);
					__m256d blend = _mm256_blendv_pd(reg_tau, reg_1, mask);
					__m256d grad = _mm256_fmadd_pd(blend, reg_rg, reg_tg);
					_mm256_storeu_pd(&p_tg[i], grad);
				}

				for (; i < n; ++i)
				{
					p_tg[i] += p_tv[i] > 0.0 ? p_rg[i] : tau * p_rg[i];
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	MathUtils::vector_sigmoid(t->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_sub(1.0, result->_data, scratch1);
				MathUtils::vector_mul(scratch1, result->_data);
				MathUtils::vector_fmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::tanh(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	MathUtils::vector_tanh(t->_data, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_sq(result->_data, scratch1);
				MathUtils::vector_sub(1.0, scratch1);
				MathUtils::vector_fmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::softmax(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	const size_t classes = t->_dimensions.back();
	const size_t batches = t->element_count() / classes;

	for (size_t b = 0; b < batches; ++b)
	{
		const size_t offset = b * classes;

		std::span<const double> t_slice(t->_data.begin() + offset, classes);

		const double max = MathUtils::vector_max(t_slice);

		const __m256d reg_max = _mm256_set1_pd(max);
		__m256d acc0 = _mm256_setzero_pd();
		__m256d acc1 = _mm256_setzero_pd();
		const double* const __restrict p_t = t_slice.data();
		double* const __restrict p_r = result->_data.data() + offset;

		size_t i = 0;
		for (; i + 8 <= classes; i += 8)
		{
			__m256d exp0 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i]), reg_max));
			__m256d exp1 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i + 4]), reg_max));

			_mm256_storeu_pd(&p_r[i], exp0);
			_mm256_storeu_pd(&p_r[i + 4], exp1);

			acc0 = _mm256_add_pd(acc0, exp0);
			acc1 = _mm256_add_pd(acc1, exp1);
		}

		__m256d acc = _mm256_add_pd(acc0, acc1);

		for (; i + 4 <= classes; i += 4)
		{
			__m256d exp = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i]), reg_max));
			_mm256_storeu_pd(&p_r[i], exp);
			acc = _mm256_add_pd(acc, exp);
		}

		double sum = MathUtils::sum_m256d(acc);

		for (; i < classes; ++i)
		{
			const double exp = std::exp(p_t[i] - max);
			p_r[i] = exp;
			sum += exp;
		}

		MathUtils::vector_div(p_r, sum, classes);
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result, batches, classes]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;

				scratch1.resize(classes);

				double* const __restrict p_tg = t->_grad.data();
				const double* const __restrict p_rv = result->_data.data();
				const double* const __restrict p_rg = result->_grad.data();

				for (size_t b = 0; b < batches; ++b)
				{
					const size_t offset = b * classes;

					const double dot = MathUtils::vector_dot(p_rg + offset, p_rv + offset, classes);
					MathUtils::vector_sub(p_rg + offset, dot, scratch1.data(), classes);
					MathUtils::vector_fmadd(p_tg + offset, p_rv + offset, scratch1.data(), classes);
				}
			};
	}

	return result;
}