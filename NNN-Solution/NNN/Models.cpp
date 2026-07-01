#include "pch.h"
#include "Models.h"
#include "Tensor.h"
#include "MathUtils.h"

void Models::soft_update(const std::vector<std::shared_ptr<Tensor>*>& agent_paras,
	const std::vector<std::shared_ptr<Tensor>*>& target_paras, double tau, double one_minus_tau)
{
	thread_local std::vector<double> scratch1;

	for (size_t i = 0; i < agent_paras.size(); ++i)
	{
		auto& agent_para = *agent_paras[i];
		auto& target_para = *target_paras[i];

		scratch1.resize(agent_para->element_count());

		MathUtils::vector_mul(target_para->data(), one_minus_tau, scratch1);
		MathUtils::vector_fmadd(scratch1, agent_para->data(), tau, target_para->mutable_data());
	}
}