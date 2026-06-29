#include "pch.h"
#include "Tensor.h"
#include "MathUtils.h"

std::shared_ptr<Tensor> Tensor::get_result_tensor(const std::shared_ptr<Tensor>& owner, const std::vector<int>& dims, bool requires_grad)
{
	owner->prepare_forward();

	const bool new_op = (int)owner->_results.size() <= owner->_op_index;

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
			const int dims_length = (int)dims.size();
			for (int i = 0; i < dims_length; ++i)
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

std::shared_ptr<Tensor> Tensor::mask_actions(const std::shared_ptr<Tensor>& q_values, const std::vector<int>& actions)
{
	const int batch_size = (int)actions.size();
	const int action_count = q_values->_dimensions.back();

	std::shared_ptr<Tensor> result = get_result_tensor(q_values, std::vector<int>{batch_size, 1}, q_values->requires_grad);

	for (int i = 0; i < batch_size; ++i)
	{
		result->_data[i] = q_values->_data[i * action_count + actions[i]];
	}

	if (!inference)
	{
		result->_parents.push_back(q_values);

		result->_backward = [q_values, actions, result, batch_size, action_count]()
			{
				if (!q_values->requires_grad) return;

				for (int i = 0; i < batch_size; ++i)
				{
					q_values->_grad[i * action_count + actions[i]] += result->_grad[i];
				}
			};
	}

	return result;
}

int Tensor::arg_max(const std::shared_ptr<const Tensor>& t)
{
	int index = 0;
	double max = t->_data[0];

	const int element_count = t->element_count();
	for (int i = 1; i < element_count; ++i)
	{
		if (t->_data[i] > max)
		{
			index = i;
			max = t->_data[i];
		}
	}

	return index;
}

std::shared_ptr<Tensor> Tensor::sum(const std::shared_ptr<Tensor>& t)
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

std::shared_ptr<Tensor> Tensor::mean(const std::shared_ptr<Tensor>& t)
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

std::shared_ptr<Tensor> Tensor::transpose(const std::shared_ptr<Tensor>& t, const std::vector<int>& axes)
{
	std::vector<int> result_dims(t->rank());
	const int axes_length = (int)axes.size();
	for (int i = 0; i < axes_length; ++i)
	{
		result_dims[i] = t->_dimensions[axes[i]];
	}

	std::shared_ptr<Tensor> result = get_result_tensor(t, result_dims, t->requires_grad);

	std::vector<int> src_indices(axes_length);
	std::vector<int> dst_indices(axes_length);

	const int element_count = t->element_count();

	for (int i = 0; i < element_count; ++i)
	{
		t->get_full_indices(i, src_indices.data());

		for (int j = 0; j < axes_length; ++j)
		{
			dst_indices[j] = src_indices[axes[j]];
		}

		result->_data[result->linear_index(dst_indices)] = t->_data[i];
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		std::vector<int> inv_axes(axes_length);
		for (int i = 0; i < axes_length; ++i)
		{
			inv_axes[axes[i]] = i;
		}

		result->_backward = [t, result, inv_axes, axes_length, element_count]()
			{
				if (!t->requires_grad) return;

				std::vector<int> src_indices(axes_length);
				std::vector<int> dst_indices(axes_length);

				for (int i = 0; i < element_count; ++i)
				{
					result->get_full_indices(i, src_indices.data());
					for (int j = 0; j < axes_length; ++j)
					{
						dst_indices[j] = src_indices[inv_axes[j]];
					}
					t->_grad[t->linear_index(dst_indices)] += result->_grad[i];
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::transpose(const std::shared_ptr<Tensor>& t)
{
	// Default permutation order: reverse all axes
	std::vector<int> axes(t->rank());
	const int axes_length = (int)axes.size();
	for (int i = 0; i < axes_length; ++i)
	{
		axes[i] = axes_length - i - 1;
	}

	return transpose(t, axes);
}

std::shared_ptr<Tensor> Tensor::broadcast(const std::shared_ptr<Tensor>& t, const std::vector<int>& target_dims)
{
	// t->_dimensions must be a suffix of target_dims (eg. t->_dimensions = [16], target_dims = [32, 16])

	std::shared_ptr<Tensor> result = get_result_tensor(t, target_dims, t->requires_grad);

	const int stride = t->element_count();
	const int blocks = result->element_count() / stride;

	for (int b = 0; b < blocks; ++b)
	{
		std::copy(t->_data.begin(), t->_data.end(), result->_data.begin() + (b * stride));
	}

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result, stride, blocks]()
			{
				if (!t->requires_grad) return;

				for (int b = 0; b < blocks; ++b)
				{
					MathUtils::vector_add(std::span<double>(t->_grad), std::span<const double>(result->_grad.data() + (b * stride), stride));
				}
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::shared_ptr<Tensor>& t, const std::vector<int>& new_dims)
{
	std::shared_ptr<Tensor> result = get_result_tensor(t, new_dims, t->requires_grad);

	result->_data = t->_data;

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result]()
			{
				if (!t->requires_grad) return;

				MathUtils::vector_add(t->_grad, result->_grad);
			};
	}

	return result;
}

std::shared_ptr<Tensor> Tensor::flatten(const std::shared_ptr<Tensor>& t, int start_axis)
{
	int flat_size = 1;
	for (int i = start_axis; i < t->rank(); ++i) flat_size *= t->_dimensions[i];

	std::vector<int> new_dims(t->_dimensions.begin(), t->_dimensions.begin() + start_axis);
	new_dims.push_back(flat_size);

	return reshape(t, new_dims);
}

std::shared_ptr<Tensor> Tensor::wrap_batch(const std::shared_ptr<Tensor>& t)
{
	std::vector<int> batch_dims;
	batch_dims.reserve(t->rank() + 1);
	batch_dims.push_back(1);
	batch_dims.insert(batch_dims.end(), t->_dimensions.begin(), t->_dimensions.end());

	std::shared_ptr<Tensor> batch = get_result_tensor(t, batch_dims, t->requires_grad);

	batch->_data = t->_data;

	if (!inference)
	{
		batch->_parents.push_back(t);

		batch->_backward = [t, batch]()
			{
				if (!t->requires_grad) return;

				MathUtils::vector_add(t->_grad, batch->_grad);
			};
	}

	return batch;
}

std::shared_ptr<Tensor> Tensor::clip(const std::shared_ptr<Tensor>& t, double min, double max)
{
	std::shared_ptr<Tensor> result = get_result_tensor(t, t->dimensions(), t->requires_grad);

	MathUtils::vector_clamp(t->_data, min, max, result->_data);

	if (!inference)
	{
		result->_parents.push_back(t);

		result->_backward = [t, result, min, max]()
			{
				if (!t->requires_grad) return;

				const size_t n = result->element_count();

				const double* __restrict p_tv = t->_data.data();
				double* __restrict p_tg = t->_grad.data();
				const __m256d reg_min = _mm256_set1_pd(min);
				const __m256d reg_max = _mm256_set1_pd(max);
				const double* __restrict p_rg = result->_grad.data();
				const __m256d reg_0 = _mm256_setzero_pd();

				size_t i = 0;
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

				for (; i < n; ++i)
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

std::shared_ptr<Tensor> Tensor::get_dense_dropout_mask(const std::vector<int>& dims, double dropout)
{
	if (inference) return std::make_shared<Tensor>(1.0, dims, false);

	const double scale = 1.0 / (1.0 - dropout);
	std::shared_ptr<Tensor> mask = std::make_shared<Tensor>(scale, dims, false);

	const int element_count = mask->element_count();
	for (int i = 0; i < element_count; ++i)
	{
		double rand = MathUtils::get_random_double();
		if (rand < dropout) mask->_data[i] = 0.0;
	}

	return mask;
}

std::shared_ptr<Tensor> Tensor::get_spatial_dropout_mask(const std::vector<int>& dims, double dropout)
{
	if (inference) return std::make_shared<Tensor>(1.0, dims, false);

	const double scale = 1.0 / (1.0 - dropout);
	std::shared_ptr<Tensor> mask = std::make_shared<Tensor>(scale, dims, false);

	const int batches = dims[0];
	const int batch_size = mask->element_count() / batches;
	const int channels = dims.back();
	const int spatial_size = batch_size / channels;

	std::vector<double> channel_vals(channels);
	for (int b = 0; b < batches; ++b)
	{
		int batch_offset = b * batch_size;

		for (int c = 0; c < channels; ++c)
		{
			double rand = MathUtils::get_random_double();
			if (rand < dropout) channel_vals[c] = 0.0;
			else channel_vals[c] = scale;
		}

		for (int s = 0; s < spatial_size; ++s)
		{
			int spatial_offset = batch_offset + (s * channels);

			for (int c = 0; c < channels; ++c)
			{
				mask->_data[spatial_offset + c] = channel_vals[c];
			}
		}
	}

	return mask;
}