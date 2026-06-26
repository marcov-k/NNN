#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

std::shared_ptr<Tensor> Tensor::get_result_tensor(std::shared_ptr<Tensor> owner, const std::vector<int>& dims, bool requires_grad)
{
	owner->prepare_forward();

	bool new_op = (int)owner->_results.size() <= owner->_op_index;

	std::shared_ptr<Tensor> result;
	if (new_op)
	{
		result = std::make_shared<Tensor>(dims, requires_grad);
		owner->_results.push_back(result);
	}
	else
	{
		result = owner->_results[owner->_op_index];

		bool shape_mismatch = result->rank() != (int)dims.size();
		if (!shape_mismatch)
		{
			for (int i = 0; i < (int)dims.size(); i++)
			{
				if (result->dimensions()[i] != dims[i])
				{
					shape_mismatch = true;
					break;
				}
			}
		}

		if (shape_mismatch)
		{
			result = std::make_shared<Tensor>(dims, requires_grad);
			owner->_results[owner->_op_index] = result;
		}
		else
		{
			result->clear_graph();
		}
	}

	owner->_op_index++;

	return result;
}

std::shared_ptr<Tensor> Tensor::sum(std::shared_ptr<Tensor> t)
{
	std::shared_ptr<Tensor> result = get_result_tensor(t, std::vector<int>(1, 1), t->requires_grad); // dims: {1} — scalar result

	result->_data[0] = MathUtils::vector_sum(t->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				MathUtils::vector_add(t->_grad, result->_grad[0]);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::mean(std::shared_ptr<Tensor> t)
{
	std::shared_ptr<Tensor> result = get_result_tensor(t, std::vector<int>(1, 1), t->requires_grad); // dims: {1} - scalar result

	result->_data[0] = MathUtils::vector_sum(t->_data) / t->element_count();

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				double rg = result->_grad[0] / (double)t->element_count();
				MathUtils::vector_add(t->_grad, rg);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::clip(std::shared_ptr<Tensor> t, double min, double max)
{
	std::shared_ptr<Tensor> result = get_result_tensor(t, t->dimensions(), t->requires_grad);

	int n = t->element_count();
	const double* __restrict p_t = t->_data.data();
	double* __restrict p_r = result->_data.data();
	__m256d reg_min = _mm256_set1_pd(min);
	__m256d reg_max = _mm256_set1_pd(max);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_t0 = _mm256_loadu_pd(&p_t[i]);
		__m256d reg_t1 = _mm256_loadu_pd(&p_t[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_t0, reg_max), reg_min);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_t1, reg_max), reg_min);

		_mm256_storeu_pd(&p_r[i], clamp0);
		_mm256_storeu_pd(&p_r[i + 4], clamp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_t = _mm256_loadu_pd(&p_t[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_t, reg_max), reg_min);
		_mm256_storeu_pd(&p_r[i], clamp);
	}

	for (; i < n; i++)
	{
		p_r[i] = std::clamp(p_t[i], min, max);
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result, min, max]()
			{
				if (!t->requires_grad) return;

				int n = t->element_count();
				const double* __restrict p_tv = t->_data.data();
				double* __restrict p_tg = t->_grad.data();
				__m256d reg_min = _mm256_set1_pd(min);
				__m256d reg_max = _mm256_set1_pd(max);
				const double* __restrict p_rg = result->_grad.data();
				__m256d reg_0 = _mm256_setzero_pd();

				int i = 0;
				for (; i <= n - 8; i += 8)
				{
					__m256d reg_tv0 = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tv1 = _mm256_loadu_pd(&p_tv[i + 4]);

					__m256d reg_tg0 = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_tg1 = _mm256_loadu_pd(&p_tg[i + 4]);

					__m256d reg_rg0 = _mm256_loadu_pd(&p_rg[i]);
					__m256d reg_rg1 = _mm256_loadu_pd(&p_rg[i + 4]);

					__m256d clamp_mask0 = _mm256_and_pd(_mm256_cmp_pd(reg_tv0, reg_max, _CMP_LE_OS), _mm256_cmp_pd(reg_tv0, reg_min, _CMP_GE_OS));
					__m256d clamp_mask1 = _mm256_and_pd(_mm256_cmp_pd(reg_tv1, reg_max, _CMP_LE_OS), _mm256_cmp_pd(reg_tv1, reg_min, _CMP_GE_OS));

					__m256d grad_add0 = _mm256_blendv_pd(reg_0, reg_rg0, clamp_mask0);
					__m256d grad_add1 = _mm256_blendv_pd(reg_0, reg_rg1, clamp_mask1);

					__m256d grad0 = _mm256_add_pd(reg_tg0, grad_add0);
					__m256d grad1 = _mm256_add_pd(reg_tg1, grad_add1);

					_mm256_storeu_pd(&p_tg[i], grad0);
					_mm256_storeu_pd(&p_tg[i + 4], grad1);
				}

				for (; i <= n - 4; i += 4)
				{
					__m256d reg_tv = _mm256_loadu_pd(&p_tv[i]);
					__m256d reg_tg = _mm256_loadu_pd(&p_tg[i]);
					__m256d reg_rg = _mm256_loadu_pd(&p_rg[i]);

					__m256d clamp_mask = _mm256_and_pd(_mm256_cmp_pd(reg_tv, reg_max, _CMP_LE_OS), _mm256_cmp_pd(reg_tv, reg_min, _CMP_GE_OS));
					
					__m256d grad_add = _mm256_blendv_pd(reg_0, reg_rg, clamp_mask);
					__m256d grad = _mm256_add_pd(reg_tg, grad_add);

					_mm256_storeu_pd(&p_tg[i], grad);
				}

				for (; i < n; i++)
				{
					if (p_tv[i] >= min && p_tv[i] <= max)
					{
						p_tg[i] += p_rg[i];
					}
				}
			};
	}

	return result;
}