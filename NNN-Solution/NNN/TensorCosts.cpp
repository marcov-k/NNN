#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

/* Cost functions - per-element - autograd graph connected */

// Computes the Mean Squared Error loss of a tensor based on the given target tensor.
std::shared_ptr<Tensor> Tensor::mse(const std::shared_ptr<Tensor>& t, const std::shared_ptr<const Tensor>& target)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Compute MSE loss -> L = (y_hat - y)^2
	MathUtils::vector_sub(t->_data, target->_data, result->_data);
	MathUtils::vector_sq(result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dL/dy_hat = 2 * (y_hat - y)
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

	// Compute pseudo-Huber loss -> L = d^2 * (sqrt(1 + ((y - y_hat) / d)^2) - 1)
	MathUtils::vector_sub(target->_data, t->_data, result->_data);
	MathUtils::vector_div(result->_data, delta);
	MathUtils::vector_sq(result->_data);
	MathUtils::vector_add(result->_data, 1.0);
	MathUtils::vector_sqrt(result->_data);
	MathUtils::vector_sub(result->_data, 1.0);
	MathUtils::vector_mul(result->_data, delta * delta);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dL/dy_hat = - (y - y_hat) / sqrt(1 + ((y - y_hat) / d)^2)
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

	std::shared_ptr<Tensor> result = get_result_tensor(t, {(int)batches, 1}, t->requires_grad);

	auto probs = std::make_shared<std::vector<double>>(element_count); // allow backward lambda to capture pointer (avoids copy allocation)

	const double* __restrict t_data = t->_data.data();
	double* __restrict p_data = probs->data();
	const double* __restrict y_data = target->_data.data();
	double* __restrict r_data = result->_data.data();

	// Compute Softmax Cross-Entropy loss per batch -> L = -sum(i = 1 to c)[y_i * ln(p_i)]; p_i = e^z_i / sum_c(e^z_c)
	for (size_t b = 0; b < batches; ++b)
	{
		const size_t offset = b * classes;

		const double* __restrict logits = t_data + offset;
		double* __restrict p = p_data + offset;

		const double max_logit = MathUtils::vector_max(logits, classes);

		const __m256d reg_max_logit = _mm256_set1_pd(max_logit);
		__m256d acc0 = _mm256_setzero_pd();
		__m256d acc1 = _mm256_setzero_pd();

		size_t i = 0;
		for (; i + 8 <= classes; i += 8)
		{
			__m256d exp0 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&logits[i]), reg_max_logit));
			__m256d exp1 = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&logits[i + 4]), reg_max_logit));

			_mm256_storeu_pd(&p[i], exp0);
			_mm256_storeu_pd(&p[i + 4], exp1);

			acc0 = _mm256_add_pd(acc0, exp0);
			acc1 = _mm256_add_pd(acc1, exp1);
		}

		__m256d acc = _mm256_add_pd(acc0, acc1);

		for (; i + 4 <= classes; i += 4)
		{
			__m256d exp = _mm256_exp_pd(_mm256_sub_pd(_mm256_loadu_pd(&logits[i]), reg_max_logit));
			_mm256_storeu_pd(&p[i], exp);
			acc = _mm256_add_pd(acc, exp);
		}

		double sum_exp = MathUtils::sum_m256d(acc);

		for (; i < classes; ++i)
		{
			double exp = std::exp(logits[i] - max_logit);
			p[i] = exp;
			sum_exp += exp;
		}

		MathUtils::vector_div(p, sum_exp, classes);

		const double* __restrict y = y_data + offset;

		size_t label = 0;
		for (size_t j = 0; j < classes; ++j)
		{
			if (y[j] > 0.5)
			{
				label = j;
				break;
			}
		}

		double loss = -std::log(p[label] + 1e-12);

		r_data[b] = loss;
	}

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dL/dz_i = p_i - y_i
		result->_backward = [t, target, result, probs, batches, classes]()
			{
				if (!t->requires_grad) return;

				thread_local std::vector<double> scratch1;
				scratch1.resize(classes);

				const double* __restrict p_data = probs->data();
				const double* __restrict y_data = target->_data.data();
				const double* __restrict r_grad = result->_grad.data();
				double* __restrict t_grad = t->_grad.data();

				// Calculate gradient per batch
				for (size_t b = 0; b < batches; ++b)
				{
					const size_t offset = b * classes;
					
					const double rg = r_grad[b];
					const double* __restrict p = p_data + offset;
					const double* __restrict y = y_data + offset;
					double* __restrict tg = t_grad + offset;

					MathUtils::vector_sub(p, y, scratch1.data(), classes);
					MathUtils::vector_fmadd(tg, scratch1.data(), rg, classes);
				}
			};
	}

	return result;
}