#pragma once

#include <cmath>
#include <memory>
#include <span>

#include "Tensor.h"

// Optimizer step function implementations.
class Optimizers
{
public:
	Optimizers() = delete;

	// Applies a Stochastic Gradient Descent optimizer step to the given parameter.
	static void sgd(const std::shared_ptr<Tensor>& para, float lr);

	// Applies an Adam optimizer step to the given parameter.
	static void adam(const std::shared_ptr<Tensor>& para, float lr, int iter, std::span<float> m, std::span<float> v,
		float beta1, float one_minus_beta1, float beta2, float one_minus_beta2, float epsilon, float weight_decay);
};