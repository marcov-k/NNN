#include "pch.h"
#include "Models.h"
#include "Tensor.h"
#include "MathUtils.h"

// Functions affecting entire neural network models.

// Clips all parameter gradients based on a maximum norm value.
void Models::clip_gradients(const std::vector<std::shared_ptr<Tensor>*>& paras, float max_norm)
{
	thread_local AlignedFloatVector scratch1;

	// Calculate total Euclidean norm of all tensors
	float total_norm = 0.0f;
	for (std::shared_ptr<Tensor>* para_ptr : paras)
	{
		auto& para = *para_ptr;
		scratch1.resize(para->element_count());
		MathUtils::vector_sq(para->grad(), scratch1);
		total_norm += MathUtils::vector_sum(scratch1);
	}
	total_norm = std::sqrt(total_norm);

	// Scale gradients to have a total Euclidean norm of max_norm -> scaling factor = max_norm / (total_norm + epsilon)
	if (total_norm > max_norm)
	{
		const float scale = max_norm / (total_norm + 1e-8f);

		for (std::shared_ptr<Tensor>* para_ptr : paras)
		{
			MathUtils::vector_mul((*para_ptr)->mutable_grad(), scale);
		}
	}
}

void Models::soft_update(const std::vector<std::shared_ptr<Tensor>*>& agent_paras,
	const std::vector<std::shared_ptr<Tensor>*>& target_paras, float tau, float one_minus_tau)
{
	thread_local AlignedFloatVector scratch1;

	// Apply soft update (Polyak averaging) to every parameter in the target model ->
	// target_param = tau * agent_param + (1 - tau) * target_param
	for (size_t i = 0; i < agent_paras.size(); ++i)
	{
		auto& agent_para = *agent_paras[i];
		auto& target_para = *target_paras[i];

		scratch1.resize(agent_para->element_count());

		MathUtils::vector_mul(target_para->data(), one_minus_tau, scratch1);
		MathUtils::vector_fmadd(scratch1, agent_para->data(), tau, target_para->mutable_data());
	}
}