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

	result->_data[0] = MathUtils::vector_sum(t->_data.data(), t->element_count());

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