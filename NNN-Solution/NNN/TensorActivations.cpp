#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

/* Activation functions - autograd graph connected */

// Applies the Rectified Linear Unit function to a tensor.
std::shared_ptr<Tensor> Tensor::relu(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Apply ReLU -> r = max(t, 0) = t (t > 0); 0 (t <= 0)
	MathUtils::vector_max(t->_data, 0.0f, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = 1 (t > 0); 0 (t <= 0)
		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				const size_t n = result->element_count();

				const float* const __restrict p_tv = t->_data.data();
				float* const __restrict p_tg = t->_grad.data();
				const float* const __restrict p_rg = result->_grad.data();
				const __m256 reg_0 = _mm256_setzero_ps();

				// AVX2 SIMD vectorize gradient calculation
				size_t i = 0;
				for (; i + 32 <= n; i += 32)
				{
					_mm256_store_ps(&p_tg[i], _mm256_add_ps(_mm256_load_ps(&p_tg[i]), _mm256_blendv_ps(reg_0, _mm256_load_ps(&p_rg[i]), _mm256_cmp_ps(_mm256_load_ps(&p_tv[i]), reg_0, _CMP_GT_OS))));
					_mm256_store_ps(&p_tg[i + 8], _mm256_add_ps(_mm256_load_ps(&p_tg[i + 8]), _mm256_blendv_ps(reg_0, _mm256_load_ps(&p_rg[i + 8]), _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 8]), reg_0, _CMP_GT_OS))));
					_mm256_store_ps(&p_tg[i + 16], _mm256_add_ps(_mm256_load_ps(&p_tg[i + 16]), _mm256_blendv_ps(reg_0, _mm256_load_ps(&p_rg[i + 16]), _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 16]), reg_0, _CMP_GT_OS))));
					_mm256_store_ps(&p_tg[i + 24], _mm256_add_ps(_mm256_load_ps(&p_tg[i + 24]), _mm256_blendv_ps(reg_0, _mm256_load_ps(&p_rg[i + 24]), _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 24]), reg_0, _CMP_GT_OS))));
				}

				for (; i + 8 <= n; i += 8)
				{
					_mm256_store_ps(&p_tg[i], _mm256_add_ps(_mm256_load_ps(&p_tg[i]), _mm256_blendv_ps(reg_0, _mm256_load_ps(&p_rg[i]), _mm256_cmp_ps(_mm256_load_ps(&p_tv[i]), reg_0, _CMP_GT_OS))));
				}

				for (; i < n; ++i)
				{
					if (p_tv[i] > 0.0f)
					{
						p_tg[i] += p_rg[i];
					}
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::leaky_relu(const std::shared_ptr<Tensor>& t, float tau)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	result->_data = t->_data;

	// Apply Leaky ReLU -> r = t (t > 0); tau * t (t <= 0)

	const size_t n = result->element_count();

	const __m256 reg_tau = _mm256_set1_ps(tau);
	float* const __restrict p_r = result->_data.data();
	const __m256 reg_0 = _mm256_setzero_ps();

	// AVX2 SIMD vectorize result calculation
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&p_r[i], _mm256_blendv_ps(_mm256_load_ps(&p_r[i]), _mm256_mul_ps(_mm256_load_ps(&p_r[i]), reg_tau), _mm256_cmp_ps(_mm256_load_ps(&p_r[i]), reg_0, _CMP_LE_OS)));
		_mm256_store_ps(&p_r[i + 8], _mm256_blendv_ps(_mm256_load_ps(&p_r[i + 8]), _mm256_mul_ps(_mm256_load_ps(&p_r[i + 8]), reg_tau), _mm256_cmp_ps(_mm256_load_ps(&p_r[i + 8]), reg_0, _CMP_LE_OS)));
		_mm256_store_ps(&p_r[i + 16], _mm256_blendv_ps(_mm256_load_ps(&p_r[i + 16]), _mm256_mul_ps(_mm256_load_ps(&p_r[i + 16]), reg_tau), _mm256_cmp_ps(_mm256_load_ps(&p_r[i + 16]), reg_0, _CMP_LE_OS)));
		_mm256_store_ps(&p_r[i + 24], _mm256_blendv_ps(_mm256_load_ps(&p_r[i + 24]), _mm256_mul_ps(_mm256_load_ps(&p_r[i + 24]), reg_tau), _mm256_cmp_ps(_mm256_load_ps(&p_r[i + 24]), reg_0, _CMP_LE_OS)));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&p_r[i], _mm256_blendv_ps(_mm256_load_ps(&p_r[i]), _mm256_mul_ps(_mm256_load_ps(&p_r[i]), reg_tau), _mm256_cmp_ps(_mm256_load_ps(&p_r[i]), reg_0, _CMP_LE_OS)));
	}
	
	for (; i < n; ++i)
	{
		if (p_r[i] <= 0.0f)
		{
			p_r[i] *= tau;
		}
	}

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = 1 (t > 0); tau (t <= 0)
		result->_backward = [t, tau, result, n]()
			{
				const float* const __restrict p_tv = t->_data.data();
				float* const __restrict p_tg = t->_grad.data();
				const __m256 reg_tau = _mm256_set1_ps(tau);
				const float* const __restrict p_rg = result->_grad.data();
				const __m256 reg_0 = _mm256_setzero_ps();
				const __m256 reg_1 = _mm256_set1_ps(1.0f);

				// AVX2 SIMD vectorize gradient calculation
				size_t i = 0;
				for (; i + 32 <= n; i += 32)
				{
					_mm256_store_ps(&p_tg[i], _mm256_fmadd_ps(_mm256_blendv_ps(reg_tau, reg_1, _mm256_cmp_ps(_mm256_load_ps(&p_tv[i]), reg_0, _CMP_GT_OS)), _mm256_load_ps(&p_rg[i]), _mm256_load_ps(&p_tg[i])));
					_mm256_store_ps(&p_tg[i + 8], _mm256_fmadd_ps(_mm256_blendv_ps(reg_tau, reg_1, _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 8]), reg_0, _CMP_GT_OS)), _mm256_load_ps(&p_rg[i + 8]), _mm256_load_ps(&p_tg[i + 8])));
					_mm256_store_ps(&p_tg[i + 16], _mm256_fmadd_ps(_mm256_blendv_ps(reg_tau, reg_1, _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 16]), reg_0, _CMP_GT_OS)), _mm256_load_ps(&p_rg[i + 16]), _mm256_load_ps(&p_tg[i + 16])));
					_mm256_store_ps(&p_tg[i + 24], _mm256_fmadd_ps(_mm256_blendv_ps(reg_tau, reg_1, _mm256_cmp_ps(_mm256_load_ps(&p_tv[i + 24]), reg_0, _CMP_GT_OS)), _mm256_load_ps(&p_rg[i + 24]), _mm256_load_ps(&p_tg[i + 24])));
				}

				for (; i + 8 <= n; i += 8)
				{
					_mm256_store_ps(&p_tg[i], _mm256_fmadd_ps(_mm256_blendv_ps(reg_tau, reg_1, _mm256_cmp_ps(_mm256_load_ps(&p_tv[i]), reg_0, _CMP_GT_OS)), _mm256_load_ps(&p_rg[i]), _mm256_load_ps(&p_tg[i])));
				}

				for (; i < n; ++i)
				{
					p_tg[i] += p_tv[i] > 0.0f ? p_rg[i] : tau * p_rg[i];
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::sigmoid(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Apply Sigmoid -> r = 1 / (1 + e^(-t))
	MathUtils::vector_sigmoid(t->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = Sigmoid(t) * (1 - Sigmoid(t))
		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				thread_local AlignedFloatVector scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_sub(1.0f, result->_data, scratch1);
				MathUtils::vector_mul(scratch1, result->_data);
				MathUtils::vector_fmadd(t->_grad, scratch1, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::tanh(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Apply Tanh -> r = (e^(2t) - 1) / (e^(2t) + 1)
	MathUtils::vector_tanh(t->_data, result->_data);

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = 1 - Tanh^2(t)
		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				thread_local AlignedFloatVector scratch1;

				const int element_count = result->element_count();
				scratch1.resize(element_count);
				MathUtils::vector_sq(result->_data, scratch1);
				MathUtils::vector_sub(1.0f, scratch1);
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

	// Apply Softmax per batch -> Softmax(z_i) = e^z_i / (sum(j = 1 to n)[e^z_j])
	for (size_t b = 0; b < batches; ++b)
	{
		const size_t offset = b * classes;

		const float* const __restrict p_t = t->_data.data() + offset;
		float* const __restrict p_r = result->_data.data() + offset;

		const float max = MathUtils::vector_max(p_t, classes);

		const __m256 reg_max = _mm256_set1_ps(max);
		__m256 acc0 = _mm256_setzero_ps();
		__m256 acc1 = _mm256_setzero_ps();
		__m256 acc2 = _mm256_setzero_ps();
		__m256 acc3 = _mm256_setzero_ps();

		// AVX2 SIMD vectorize result calculation
		size_t i = 0;
		for (; i + 32 <= classes; i += 32)
		{
			const __m256 exp0 = _mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(&p_t[i]), reg_max));
			const __m256 exp1 = _mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(&p_t[i + 8]), reg_max));
			const __m256 exp2 = _mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(&p_t[i + 16]), reg_max));
			const __m256 exp3 = _mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(&p_t[i + 24]), reg_max));

			_mm256_store_ps(&p_r[i], exp0);
			_mm256_store_ps(&p_r[i + 8], exp1);
			_mm256_store_ps(&p_r[i + 16], exp2);
			_mm256_store_ps(&p_r[i + 24], exp3);

			acc0 = _mm256_add_ps(acc0, exp0);
			acc1 = _mm256_add_ps(acc1, exp1);
			acc2 = _mm256_add_ps(acc2, exp2);
			acc3 = _mm256_add_ps(acc3, exp3);
		}

		acc0 = _mm256_add_ps(acc0, acc1);
		acc2 = _mm256_add_ps(acc2, acc3);
		acc0 = _mm256_add_ps(acc0, acc2);

		for (; i + 8 <= classes; i += 8)
		{
			const __m256 exp = _mm256_exp_ps(_mm256_sub_ps(_mm256_load_ps(&p_t[i]), reg_max));
			_mm256_store_ps(&p_r[i], exp);
			acc0 = _mm256_add_ps(acc0, exp);
		}

		float sum = MathUtils::sum_m256(acc0);

		for (; i < classes; ++i)
		{
			const float exp = std::exp(p_t[i] - max);
			p_r[i] = exp;
			sum += exp;
		}

		// Normalize result values
		MathUtils::vector_div(p_r, sum, classes);
	}

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> grad_t_i = r_i * (grad_r_i - sum_j(grad_r_j * r_j))
		result->_backward = [t, result, batches, classes]()
			{
				if (!t->requires_grad) return;

				thread_local AlignedFloatVector scratch1;

				scratch1.resize(classes);

				float* const __restrict p_tg = t->_grad.data();
				const float* const __restrict p_rv = result->_data.data();
				const float* const __restrict p_rg = result->_grad.data();

				// Compute gradients per batch
				for (size_t b = 0; b < batches; ++b)
				{
					const size_t offset = b * classes;

					const float dot = MathUtils::vector_dot(p_rg + offset, p_rv + offset, classes);
					MathUtils::vector_sub(p_rg + offset, dot, scratch1.data(), classes);
					MathUtils::vector_fmadd(p_tg + offset, p_rv + offset, scratch1.data(), classes);
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::linear(const std::shared_ptr<Tensor>& t)
{
	auto result = get_result_tensor(t, t->_dimensions, t->requires_grad);

	// Copy exact data - linear function
	std::copy(t->_data.begin(), t->_data.end(), result->_data.begin());

	// Connect result tensor to autograd graph if needed
	if (!inference)
	{
		result->_parents.push_back(t);

		// Gradient calculation function -> dr/dt = 1
		result->_backward = [t, result]()
			{
				MathUtils::vector_add(t->_grad, result->_grad);
			};
	}

	return result;
}