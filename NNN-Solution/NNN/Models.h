#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "Tensor.h"

// Functions affecting entire neural network models.
class Models
{
public:
	Models() = delete;

	// Clips all parameter gradients based on a maximum norm value.
	static void clip_gradients(const std::vector<std::shared_ptr<Tensor>*>& paras, double max_norm);

	// Performs a soft update on the parameters of the target model given the parameters of the agent model (DQN).
	static void soft_update(const std::vector<std::shared_ptr<Tensor>*>& agent_paras,
		const std::vector<std::shared_ptr<Tensor>*>& target_paras, double tau, double one_minus_tau);
};