#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

std::shared_ptr<Tensor> Tensor::mse(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	MathUtils::vector_sub(t->_data, target->_data, result->_data);
	MathUtils::vector_sq(result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, target, result]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_sub(t->_data, target->_data, scratch1);
				MathUtils::vector_mul(scratch1, 2.0);
				MathUtils::vector_fmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::huber(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target, double delta)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	MathUtils::vector_sub(target->_data, t->_data, result->_data);
	MathUtils::vector_div(result->_data, delta);
	MathUtils::vector_sq(result->_data);
	MathUtils::vector_add(result->_data, 1.0);
	MathUtils::vector_sqrt(result->_data);
	MathUtils::vector_sub(result->_data, 1.0);
	MathUtils::vector_mul(result->_data, delta * delta);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, target, delta, result]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;
				thread_local std::vector<double> scratch2;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				scratch2.resize(element_count);

				MathUtils::vector_sub(target->_data, t->_data, scratch1);
				MathUtils::vector_div(scratch1, delta, scratch2);
				MathUtils::vector_sq(scratch2);
				MathUtils::vector_add(scratch2, 1.0);
				MathUtils::vector_sqrt(scratch2);
				MathUtils::vector_div(scratch1, scratch2);
				MathUtils::vector_fnmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::softmax_cross_entropy(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target)
{
	const size_t classes = t->_dimensions.back();
	const size_t element_count = t->element_count();
	const size_t batches = element_count / classes;

	std::shared_ptr<Tensor> result = get_result_tensor(t, std::vector<int>{(int)batches, 1}, t->requires_grad);

	std::vector<double> probs(element_count);

	for (int b = 0; b < batches; ++b)
	{
		const size_t offset = b * classes;

		std::span<const double> t_slice(t->_data.begin() + offset, classes);

		const double max = MathUtils::vector_max(t_slice);

		std::span<double> p_slice(probs.begin() + offset, classes);

		const __m256d reg_max = _mm256_set1_pd(max);
		__m256d acc0 = _mm256_setzero_pd();
		__m256d acc1 = _mm256_setzero_pd();
		const double* const __restrict p_t = t_slice.data();
		double* const __restrict p_p = p_slice.data();

		size_t i = 0;
		for (; i <= classes - 8; i += 8)
		{
			__m256d exp0 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i]), reg_max));
			__m256d exp1 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i + 4]), reg_max));

			_mm256_storeu_pd(&p_p[i], exp0);
			_mm256_storeu_pd(&p_p[i + 4], exp1);

			acc0 = _mm256_add_pd(acc0, exp0);
			acc1 = _mm256_add_pd(acc1, exp1);
		}

		__m256d acc = _mm256_add_pd(acc0, acc1);

		for (; i <= classes - 4; i += 4)
		{
			__m256d exp = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&p_t[i]), reg_max));
			_mm256_storeu_pd(&p_p[i], exp);
			acc = _mm256_add_pd(acc, exp);
		}

		double sum_exp = MathUtils::sum_m256d(acc);

		for (; i < classes; ++i)
		{
			double exp = std::exp(p_t[i] - max);
			p_p[i] = exp;
			sum_exp += exp;
		}

		const double log_sum_exp = std::log(sum_exp) + max;

		MathUtils::vector_div(p_slice, sum_exp);

		std::span<const double> g_slice(target->_data.begin() + offset, classes);

		const double dot = MathUtils::vector_dot(g_slice, t_slice);

		result->_data[b] = log_sum_exp - dot;
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, target, result, probs, batches, classes]()
			{
				if (!t->requires_grad) return;

				for (size_t b = 0; b < batches; ++b)
				{
					const size_t offset = b * classes;

					const double r_grad = result->_grad[b];

					std::span<const double> p_slice(probs.begin() + offset, classes);
					std::span<const double> g_slice(target->_data.begin() + offset, classes);
					std::span<double> tg_slice(t->_grad.begin() + offset, classes);

					const double* const __restrict p_p = p_slice.data();
					const double* const __restrict p_g = g_slice.data();
					double* const __restrict p_tg = tg_slice.data();
					const __m256d reg_r_grad = _mm256_set1_pd(r_grad);

					int i = 0;
					for (; i <= classes - 8; i += 8)
					{
						__m256d reg_p0 = _mm256_loadu_pd(&p_p[i]);
						__m256d reg_p1 = _mm256_loadu_pd(&p_p[i + 4]);

						__m256d reg_g0 = _mm256_loadu_pd(&p_g[i]);
						__m256d reg_g1 = _mm256_loadu_pd(&p_g[i + 4]);

						__m256d reg_tg0 = _mm256_loadu_pd(&p_tg[i]);
						__m256d reg_tg1 = _mm256_loadu_pd(&p_tg[i + 4]);

						__m256d diff0 = _mm256_sub_pd(reg_p0, reg_g0);
						__m256d diff1 = _mm256_sub_pd(reg_p1, reg_g1);

						__m256d res0 = _mm256_fmadd_pd(diff0, reg_r_grad, reg_tg0);
						__m256d res1 = _mm256_fmadd_pd(diff1, reg_r_grad, reg_tg1);

						_mm256_storeu_pd(&p_tg[i], res0);
						_mm256_storeu_pd(&p_tg[i + 4], res1);
					}

					for (; i <= classes - 4; i += 4)
					{
						__m256d reg_p = _mm256_loadu_pd(&p_p[i]);
						__m256d reg_g = _mm256_loadu_pd(&p_g[i]);
						__m256d reg_tg = _mm256_loadu_pd(&p_tg[i]);
						__m256d diff = _mm256_sub_pd(reg_p, reg_g);
						__m256d res = _mm256_fmadd_pd(diff, reg_r_grad, reg_tg);
						_mm256_storeu_pd(&p_tg[i], res);
					}

					for (; i < classes; ++i)
					{
						tg_slice[i] += (p_slice[i] - g_slice[i]) * r_grad;
					}
				}
			};
	}

	return result;
}