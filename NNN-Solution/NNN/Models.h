#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "Tensor.h"

class Models
{
public:
	Models() = delete;

	static void soft_update(const std::vector<std::shared_ptr<Tensor>*>& agent_paras,
		const std::vector<std::shared_ptr<Tensor>*>& target_paras, double tau, double one_minus_tau);
};